"""
FollowUp Companion -- Real CALL-E Pipeline
==========================================

Initiates a real outbound call via the CALL-E API, polls until the
transcript arrives, then runs the complete FollowUp Companion pipeline:

  [0]  Pre-call preparation    ProfileStore + CallHistoryRAG + CallPlanner
  [1]  Call initiation         POST /v1/calls via CalleClient
  [2]  Call polling            GET /v1/calls/{id} until status=completed
  [3]  LLM field extraction    RealExtractor (gpt-4o-mini) or MockExtractor
  [4]  Arbitration gate        7-dimension evaluation
  [5]  Triage decision         ACCEPT / TRIAGE_RISK / TRIAGE_AMBIGUOUS
  [6]  Post-call memory        ProfileStore.update + CallHistoryRAG

Requirements:
  CALLE_API_KEY   -- set in .env or shell environment
  OPENAI_API_KEY  -- for real extraction (--extractor real, default)

Usage:
  python demo/run_real_call.py +12024406665
  python demo/run_real_call.py +12024406665 --patient-id PAT-001
  python demo/run_real_call.py +12024406665 --script-id SCRIPT-FOLLOWUP-EN
  python demo/run_real_call.py +12024406665 --poll-interval 20 --timeout 900
  python demo/run_real_call.py +12024406665 --extractor mock   # dry-run: skip real API
"""
from __future__ import annotations

import argparse
import sys
import time
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.models import CallPlan
from agents.symptom_probe import PROBE_REGISTRY, SymptomCategory
from arbitration import ArbitrationGate
from arbitration.models import Action, ArbitrationInput, CallSignals, DimensionName
from memory.call_history_rag import CallHistoryRAG
from memory.profile_store import ProfileStore
from providers.calle_client import CalleAPIError, CalleClient, CallRequest, CallResult

# ---------------------------------------------------------------------------
# Re-use display helpers and stage runners from run_call.py
# ---------------------------------------------------------------------------
# Import carefully: only functions that don't depend on the DEMO_CALLS dict.
from demo.run_call import (  # noqa: E402
    DEFAULT_CALL_SIGNALS,
    DIMENSION_LABELS,
    DIMENSION_ORDER,
    W,
    _fail,
    _header,
    _kv,
    _ok,
    _rule,
    _stage,
    _warn,
    _make_extractor,
    _try_make_planner,
    _run_stage2_extraction,
    _run_stage3_gate,
    _run_stage4_decision,
    _run_stage5_memory,
    _run_stage5_confirmation_probe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCRIPT_ID    = "SCRIPT-FOLLOWUP-EN"
TERMINAL_STATUSES    = {"completed", "failed", "canceled", "cancelled", "no-answer", "error"}
CALL_IN_PROGRESS     = {"in_progress", "ringing", "answered"}


# ---------------------------------------------------------------------------
# Task string builder  (CallPlan -> CALL-E task instruction)
# ---------------------------------------------------------------------------

def _build_task_from_plan(
    plan: CallPlan,
    patient_id: str,
    script_id: str,
) -> str:
    """
    Convert a CallPlanner plan into the natural-language task string that
    CALL-E's conversational AI will follow during the live call.

    Three sections injected directly from the plan:
      1. Opening line (plan.opening)
      2. Ordered question list (plan.questions, priority-sorted)
      3. Tone / pacing directives (plan.tone_directives)

    A fourth section encodes the symptom probe rules from PROBE_REGISTRY so
    CALL-E will ask location / quality / duration follow-ups whenever the
    patient mentions pain, breathlessness, or a wound concern.  These rules
    are derived from the same checklist that the post-call analysis uses, so
    the in-call probing and the post-call arbitration stay in sync.
    """
    SEP = "-" * 52
    lines: list[str] = [
        f"You are conducting a post-care telephone follow-up call for patient {patient_id}.",
        f"(Script reference: {script_id})",
        "",
        f"-- OPENING {SEP}",
        plan.opening,
        "",
        f"-- TONE AND PACING (mandatory) {SEP}",
    ]
    for d in plan.tone_directives:
        lines.append(f"- {d}")

    lines += [
        "",
        f"-- QUESTIONS (ask every item; priority 1 first) {SEP}",
    ]
    sorted_qs = sorted(plan.questions, key=lambda q: (q.priority, q.field_name))
    for i, q in enumerate(sorted_qs, 1):
        lines.append(f"{i}. [{q.field_name}]  {q.question}")

    # Symptom probe rules — pulled from the same PROBE_REGISTRY used by the
    # post-call analysis so in-call behaviour and scoring stay in sync.
    lines += [
        "",
        f"-- SYMPTOM PROBING (required when a symptom is reported) {SEP}",
        "If the patient reports pain, aching, discomfort, or any physical",
        "symptom, ask ALL of the following follow-up questions before moving",
        "to the next topic:",
    ]
    for dim in PROBE_REGISTRY[SymptomCategory.PAIN]:
        if dim.required:
            lines.append(f'  -> "{dim.question}"')

    lines += [
        "",
        "If the patient reports a breathing difficulty (cough, breathlessness,",
        "chest tightness), ask:",
    ]
    for dim in PROBE_REGISTRY[SymptomCategory.RESPIRATORY]:
        if dim.required:
            lines.append(f'  -> "{dim.question}"')

    lines += [
        "",
        "If the patient mentions their wound, incision, or surgical site, ask:",
    ]
    for dim in PROBE_REGISTRY[SymptomCategory.WOUND]:
        if dim.required:
            lines.append(f'  -> "{dim.question}"')

    lines += [
        "",
        f"-- CLOSING {SEP}",
        plan.closing,
    ]
    return "\n".join(lines)


def _build_result_schema(plan: CallPlan) -> dict:
    """
    Build a CALL-E result_schema that requests the verbatim transcript plus
    structured symptom-probe fields.  The structured fields let us validate
    that CALL-E actually collected the probe answers even before our extractor
    runs.
    """
    return {
        "type": "object",
        "required": ["transcript"],
        "properties": {
            "transcript": {
                "type": "string",
                "description": (
                    "Complete verbatim transcript.  Each turn on its own line "
                    "as 'Agent: ...' or 'Patient: ...'.  Mark inaudible "
                    "segments as [unintelligible]."
                ),
            },
            "call_answered": {
                "type": "boolean",
                "description": "True if the patient answered the call.",
            },
            "duration_seconds": {
                "type": "number",
                "description": "Approximate call duration in seconds.",
            },
            # Symptom probe fields — populated only when the patient reported pain.
            "pain_location": {
                "type": "string",
                "description": (
                    "Body location of reported pain (e.g. 'left upper arm', "
                    "'chest').  Empty string if no pain was reported."
                ),
            },
            "pain_quality": {
                "type": "string",
                "description": (
                    "Character of reported pain (e.g. 'sharp', 'dull', "
                    "'burning').  Empty string if not asked or not reported."
                ),
            },
            "pain_duration": {
                "type": "string",
                "description": (
                    "How long the pain has been present.  Empty string if not "
                    "asked or not reported."
                ),
            },
        },
    }

# ---------------------------------------------------------------------------
# Stage 0 – Pre-call preparation
# ---------------------------------------------------------------------------

def _run_stage0_preplan(
    patient_id: str,
    phone_number: str,
    planner,
    profile_store: ProfileStore,
    rag: CallHistoryRAG,
) -> "CallPlan | None":
    """
    Generate a pre-call plan and return it so stage 1 can embed it in the
    CALL-E task string.  Returns None when no planner is available.
    """
    t0 = time.perf_counter()
    profile = profile_store.get_profile(patient_id)
    history = rag.retrieve_recent(patient_id, n=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _stage(0, "Pre-call preparation", elapsed_ms)
    _kv("Patient",       patient_id)
    _kv("Phone",         phone_number)
    _kv("Call history",  f"{rag.count(patient_id)} call(s) in memory")
    _kv("Profile conf.", f"{profile.confidence:.2f} ({profile.call_count} calls)")

    if history:
        print("      Recent history:")
        for r in history[:3]:
            print(f"        {r.call_id}  {r.timestamp[:10]}  "
                  f"topics: {', '.join(r.topics or ['-'])}")

    if planner is None:
        _warn("OPENAI_API_KEY not set -- skipping call plan generation")
        return None

    t1 = time.perf_counter()
    try:
        plan = planner.plan_call(patient_id)
        plan_ms = (time.perf_counter() - t1) * 1000
        mode = "personalised" if plan.personalization_used else "neutral (default)"
        fallback = "  [FALLBACK]" if plan.fallback_used else ""
        print(f"      [CallPlanner] plan generated in {plan_ms:.0f} ms ({mode}){fallback}")
        _kv("Opening", f'"{plan.opening[:68]}..."' if len(plan.opening) > 68 else f'"{plan.opening}"')
        print("      Questions:")
        for q in plan.questions[:5]:
            flag = " [history]" if q.from_history else ""
            print(f"        P{q.priority}  {q.field_name:<24} {q.question[:52]}{flag}")
        print(f"      [Probe rules] PAIN ({sum(1 for d in PROBE_REGISTRY[SymptomCategory.PAIN] if d.required)} required), "
              f"RESPIRATORY ({sum(1 for d in PROBE_REGISTRY[SymptomCategory.RESPIRATORY] if d.required)} required), "
              f"WOUND ({sum(1 for d in PROBE_REGISTRY[SymptomCategory.WOUND] if d.required)} required) "
              f"-> will be embedded in CALL-E task")
        return plan
    except Exception as exc:
        _warn(f"CallPlanner error (continuing anyway): {exc}")
        return None


# ---------------------------------------------------------------------------
# Stage 1 – Initiate the real call
# ---------------------------------------------------------------------------

def _run_stage1_initiate(
    client: CalleClient,
    patient_id: str,
    phone_number: str,
    script_id: str,
    plan: "CallPlan | None" = None,
) -> CallResult:
    """
    POST /v1/calls and return the initial CallResult (status=queued).

    When a CallPlan is provided, builds a rich task string from:
      - plan.opening          -- how the agent greets the patient
      - plan.questions        -- the ordered question list
      - plan.tone_directives  -- pacing instructions
      - PROBE_REGISTRY rules  -- conditional symptom follow-ups

    Without a plan, falls back to the generic _TASK_TEMPLATE.
    """
    _stage(1, "Initiating outbound call via CALL-E")

    if plan is not None:
        task_desc   = _build_task_from_plan(plan, patient_id, script_id)
        result_sch  = _build_result_schema(plan)
        task_source = "CallPlanner (personalised + probe rules)"
    else:
        task_desc   = None   # calle_client.py will use _TASK_TEMPLATE
        result_sch  = None   # calle_client.py will use _TRANSCRIPT_RESULT_SCHEMA
        task_source = "generic fallback template"

    req = CallRequest(
        patient_id=patient_id,
        phone_number=phone_number,
        call_script_id=script_id,
        transcription_language="en-US",
        metadata={"source": "followup-companion-demo", "demo": True},
        task_description=task_desc,
        result_schema=result_sch,
    )

    _kv("Endpoint",     f"{client.base_url}/calls")
    _kv("Patient",      req.patient_id)
    _kv("Phone",        req.phone_number)
    _kv("Script ID",    req.call_script_id)
    _kv("Language",     req.transcription_language)
    _kv("Task source",  task_source)
    if task_desc:
        # Show a compact preview of the task (first 3 lines)
        preview_lines = task_desc.splitlines()[:3]
        _kv("Task preview", preview_lines[0][:60])
        for line in preview_lines[1:3]:
            print(f"{'':34}{line[:60]}")

    t0 = time.perf_counter()
    result = client.initiate_call(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _ok(f"Call queued  ({elapsed_ms:.0f} ms)")
    _kv("Call ID",   result.call_id)
    _kv("Status",    result.status)
    return result


# ---------------------------------------------------------------------------
# Stage 2 – Poll until the call completes and has a transcript
# ---------------------------------------------------------------------------

def _run_stage2_poll(
    client: CalleClient,
    call_id: str,
    poll_interval_s: float,
    timeout_s: float,
) -> CallResult:
    """
    Poll GET /v1/calls/{call_id} until terminal status.

    Returns the completed CallResult.
    Raises SystemExit if timeout is exceeded or the call fails.
    """
    _stage(2, f"Polling for transcript  (interval={poll_interval_s:.0f}s  timeout={timeout_s:.0f}s)")

    deadline = time.monotonic() + timeout_s
    prev_status = ""
    poll_n = 0

    while time.monotonic() < deadline:
        poll_n += 1
        t0 = time.perf_counter()
        try:
            result = client.get_call_status(call_id)
        except CalleAPIError as exc:
            if exc.is_retryable:
                _warn(f"Transient error on poll #{poll_n}: {exc}  (retrying)")
                time.sleep(min(poll_interval_s, 10))
                continue
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        status = result.status

        if status != prev_status:
            ts = time.strftime("%H:%M:%S")
            print(f"      [{ts}]  poll #{poll_n:>3}  status={status:<14}  ({elapsed_ms:.0f} ms)")
            prev_status = status

        if status in TERMINAL_STATUSES:
            print()
            if status == "completed":
                _ok(f"Call completed  (call_id={call_id})")
                if result.duration_s is not None:
                    _kv("Duration", f"{result.duration_s:.0f} s")
                if result.transcript:
                    words = len(result.transcript.split())
                    unintelligible = result.transcript.count("[unintelligible]")
                    _kv("Transcript", f"{words} words  |  {unintelligible} [unintelligible] markers")
                else:
                    _warn("Call completed but transcript is empty -- pipeline will proceed with empty text")
                return result
            else:
                _fail(f"Call ended with non-completable status: {status!r}")
                raise SystemExit(f"Call {call_id!r} ended with status {status!r} -- no transcript to process.")

        remaining = deadline - time.monotonic()
        sleep_for = min(poll_interval_s, remaining)
        if sleep_for > 0:
            time.sleep(sleep_for)

    _fail(f"Timeout after {timeout_s:.0f}s waiting for call {call_id!r} to complete.")
    raise SystemExit(f"Poll timeout exceeded. Last known call_id={call_id!r}  status={prev_status!r}.\n"
                     f"Re-run with a longer --timeout or check the CALL-E dashboard.")


# ---------------------------------------------------------------------------
# Stage 3 – Print transcript
# ---------------------------------------------------------------------------

def _run_stage3_transcript(result: CallResult) -> None:
    _stage(3, "Transcript received")
    _kv("Call ID",  result.call_id)
    _kv("Status",   result.status)
    if result.duration_s is not None:
        _kv("Duration", f"{result.duration_s:.0f} s")
    if result.recording_url:
        _kv("Recording", result.recording_url)

    transcript = result.transcript or ""
    words = len(transcript.split()) if transcript else 0
    unintelligible = transcript.count("[unintelligible]") if transcript else 0
    _kv("Transcript", f"{words} words  |  {unintelligible} [unintelligible] markers")
    print()

    lines = transcript.splitlines() if transcript else ["(no transcript)"]
    for line in lines[:14]:
        print(f"      {line}")
    if len(lines) > 14:
        print(f"      ... ({len(lines) - 14} more lines)")


# ---------------------------------------------------------------------------
# Build a scenario dict compatible with the shared stage runners
# ---------------------------------------------------------------------------

def _call_result_to_scenario(
    result: CallResult,
    patient_id: str,
    script_id: str,
) -> dict:
    """
    Convert a CALL-E CallResult into the scenario dict format used by the
    shared stage runners (_run_stage2_extraction, etc.).
    """
    transcript = result.transcript or ""
    return {
        "scenario_id":      result.call_id,
        "category":         "real_call",
        "description":      f"Live call to patient {patient_id} via script {script_id}",
        "transcript":       transcript,
        "expected_action":  None,               # not known for live calls
        "mock_llm_evidence": [],                # unused when extractor=real
        "mock_raw_llm_output": {},              # unused when extractor=real
        "summary_for_rag":  f"Real call {result.call_id}. Duration: "
                            f"{result.duration_s or '?'} s. "
                            f"Transcript: {len(transcript.split())} words.",
        "rag_topics":       ["real_call"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Reconfigure stdout to UTF-8 so Chinese characters survive on Windows GBK.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Initiate a real CALL-E outbound call and run the full FollowUp pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phone_number",
        help="E.164 phone number to call, e.g. +12024406665",
    )
    parser.add_argument(
        "--patient-id", default=None, metavar="ID",
        help="Patient identifier (default: derived from phone number).",
    )
    parser.add_argument(
        "--script-id", default=DEFAULT_SCRIPT_ID, metavar="ID",
        help=f"CALL-E script template ID (default: {DEFAULT_SCRIPT_ID}).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30.0, metavar="SECS",
        help="Seconds between status polls while call is in progress (default: 30).",
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, metavar="SECS",
        help="Max seconds to wait for the call to complete before giving up (default: 900).",
    )
    parser.add_argument(
        "--extractor", choices=["mock", "real"], default="real",
        help="LLM extractor (default: real; needs OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    phone = args.phone_number
    patient_id = args.patient_id or f"PAT-{phone.lstrip('+').replace(' ', '')[-7:]}"

    # -- Shared resources --------------------------------------------------
    profile_store = ProfileStore()
    rag           = CallHistoryRAG(persist_dir=None, collection_name="real_call_run")
    gate          = ArbitrationGate()
    extractor     = _make_extractor(args.extractor)
    planner       = _try_make_planner(rag, profile_store)

    _header("FOLLOWUP COMPANION -- REAL CALL PIPELINE", char="#")
    print(f"  Phone       : {phone}")
    print(f"  Patient     : {patient_id}")
    print(f"  Script ID   : {args.script_id}")
    print(f"  Extractor   : {args.extractor}")
    print(f"  CallPlanner : {'real (gpt-4o-mini)' if planner else 'skipped (no API key)'}")
    print(f"  Poll        : every {args.poll_interval:.0f}s, timeout {args.timeout:.0f}s")

    wall_t0 = time.perf_counter()

    # -- Stage 0: Pre-call -------------------------------------------------
    plan = _run_stage0_preplan(patient_id, phone, planner, profile_store, rag)

    # -- Stage 1: Initiate real call via CALL-E ----------------------------
    try:
        client = CalleClient()
    except EnvironmentError as exc:
        _fail(str(exc))
        raise SystemExit(1) from exc

    try:
        queued_result = _run_stage1_initiate(client, patient_id, phone, args.script_id, plan)
    except CalleAPIError as exc:
        print()
        _fail(f"CALL-E API error during call initiation:")
        _fail(f"  status_code : {exc.status_code}")
        _fail(f"  message     : {exc}")
        _fail(f"  retryable   : {exc.is_retryable}")
        if exc.status_code == 404:
            print()
            print("  NOTE: /v1/calls returned 404.")
            print("        Check that CALLE_BASE_URL points to the correct environment.")
            print(f"        Current base URL: {client.base_url}")
        raise SystemExit(1) from exc

    call_id = queued_result.call_id

    # -- Stage 2: Poll until transcript arrives ----------------------------
    completed_result = _run_stage2_poll(
        client, call_id,
        poll_interval_s=args.poll_interval,
        timeout_s=args.timeout,
    )

    # -- Stage 3: Show transcript ------------------------------------------
    _run_stage3_transcript(completed_result)

    # -- Build scenario dict for shared stage runners ----------------------
    scenario = _call_result_to_scenario(completed_result, patient_id, args.script_id)

    # -- Stage 4 (numbered 4 in user-visible output): extraction ----------
    # Reuse shared runner; it labels itself "[2] LLM field extraction" internally.
    # We offset numbering here for clarity in the header line only.
    llm_evidence, raw_output, _ext_ms = _run_stage2_extraction(scenario, extractor)

    # -- Stage 5: Arbitration gate ----------------------------------------
    result, _gate_ms = _run_stage3_gate(scenario, llm_evidence, raw_output, gate)

    # -- Stage 6: Triage decision -----------------------------------------
    _run_stage4_decision(result)

    # -- Stage 7: Post-call memory ----------------------------------------
    _run_stage5_memory(patient_id, scenario, result, profile_store, rag)

    # -- In-call agent layer ----------------------------------------------
    _run_stage5_confirmation_probe(llm_evidence)

    # -- Summary ----------------------------------------------------------
    wall_ms = (time.perf_counter() - wall_t0) * 1000
    action  = result.action.value.upper()

    print()
    _rule("=")
    print(f"  CALL ID      : {call_id}")
    print(f"  PATIENT      : {patient_id}")
    print(f"  PHONE        : {phone}")
    print(f"  TRIAGE       : {action}")
    print(f"  WALL TIME    : {wall_ms / 1000:.1f} s  (excl. call duration)")
    _rule("=")

    # Token usage
    usage = extractor.token_usage
    if usage:
        print()
        _rule("-")
        print("  TOKEN USAGE (extractor)")
        _kv("Model",          usage["model"])
        _kv("API calls",      str(usage["calls"]))
        _kv("Total tokens",
            f"{usage['total_tokens']:,}  "
            f"(prompt {usage['prompt_tokens']:,}  "
            f"completion {usage['completion_tokens']:,})")
        _kv("Estimated cost", f"${usage['estimated_cost_usd']:.4f} USD")
        _rule("-")


if __name__ == "__main__":
    main()
