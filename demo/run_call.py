"""
FollowUp Companion -- End-to-End Pipeline Demo
==============================================

Exercises all six modules in sequence for a demo patient across three
representative calls, one per triage outcome:

  CALL-DEMO-A  Routine post-op check           -> ACCEPT
  CALL-DEMO-B  Behavioral farewell indicators  -> TRIAGE_RISK
  CALL-DEMO-C  Degraded transcript             -> TRIAGE_AMBIGUOUS

Pipeline stages for each call
------------------------------
  [0] Pre-call preparation   ProfileStore + CallHistoryRAG + CallPlanner
  [1] Transcript received    Simulated CALL-E webhook delivery
  [2] LLM field extraction   RealExtractor (gpt-4o-mini) or MockExtractor
  [3] Arbitration gate       7-dimension evaluation
  [4] Triage decision        Outcome + human-readable message
  [5] Post-call memory       ProfileStore.update + CallHistoryRAG.add_call_summary

Usage
-----
  python demo/run_call.py                     # real extractor (needs OPENAI_API_KEY)
  python demo/run_call.py --extractor mock    # no API key, pre-baked evidence
  python demo/run_call.py --scenario SC-004   # single scenario from eval JSONL
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap (allows running from repo root or demo/ directly)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arbitration import ArbitrationGate
from arbitration.models import Action, ArbitrationInput, CallSignals, DimensionName
from agents import (
    ConfirmationTracker,
    build_probe_plan,
    classify_ambiguity,
    is_symptom_incomplete,
    AmbiguityKind,
)
from agents.models import STANDARD_PROTOCOL_ITEMS
from memory.call_history_rag import CallHistoryRAG
from memory.profile_store import ProfileStore

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

W = 72   # column width

def _rule(char: str = "-") -> None:
    print(char * W)

def _header(title: str, char: str = "=") -> None:
    print()
    print(char * W)
    print(f"  {title}")
    print(char * W)

def _stage(n: int, label: str, elapsed_ms: float | None = None) -> None:
    suffix = f"  [{elapsed_ms:.1f} ms]" if elapsed_ms is not None else ""
    print(f"\n  [{n}] {label}{suffix}")
    print(f"  {'-' * (W - 4)}")

def _kv(key: str, value: str, indent: int = 6) -> None:
    pad = " " * indent
    key_str = f"{key}:".ljust(28)
    print(f"{pad}{key_str}{value}")

def _block(label: str, text: str, indent: int = 6, width: int = 62) -> None:
    pad = " " * indent
    print(f"{pad}{label}:")
    for line in textwrap.wrap(text, width=width):
        print(f"{pad}  {line}")

def _ok(msg: str)  -> None: print(f"      OK  {msg}")
def _warn(msg: str) -> None: print(f"    WARN  {msg}")
def _fail(msg: str) -> None: print(f"    FAIL  {msg}")


DIMENSION_ORDER = [
    DimensionName.PII_BOUNDARY,
    DimensionName.TRANSCRIPTION_QUALITY,
    DimensionName.RISK_SIGNAL,
    DimensionName.OOD,
    DimensionName.COMPLETENESS,
    DimensionName.TRACEABILITY,
    DimensionName.CONFIDENCE,
]

DIMENSION_LABELS = {
    DimensionName.PII_BOUNDARY:          "PII Boundary",
    DimensionName.TRANSCRIPTION_QUALITY: "Transcription Quality",
    DimensionName.RISK_SIGNAL:           "Risk Signal",
    DimensionName.OOD:                   "OOD / In-Domain",
    DimensionName.COMPLETENESS:          "Completeness",
    DimensionName.TRACEABILITY:          "Traceability",
    DimensionName.CONFIDENCE:            "Confidence",
}

# ---------------------------------------------------------------------------
# Demo scenarios  (inline; do not require OPENAI for the transcript itself)
# ---------------------------------------------------------------------------

#  Three scenarios covering all three triage outcomes.
#  When --extractor real is used the LLM re-extracts from the transcript.
#  When --extractor mock the pre-baked evidence below is used instead.

DEMO_CALLS: list[dict] = [
    # -- CALL-DEMO-A: routine post-op, all clear ------------------------
    {
        "scenario_id": "CALL-DEMO-A",
        "category": "normal",
        "description": "Routine week-4 post-op check -- no concerns",
        "transcript": (
            "Agent: Hello, this is your follow-up team calling. Am I speaking "
            "with the patient?\n"
            "Patient: Yes, this is me. Hello.\n"
            "Agent: How have you been feeling since the surgery?\n"
            "Patient: Much better, thank you. The incision is healing well. "
            "I have no new symptoms.\n"
            "Agent: Good to hear. Are you taking your medication as prescribed?\n"
            "Patient: Yes, I take all my tablets every morning without missing a dose.\n"
            "Agent: On a scale of 1 to 10, how would you rate your pain today?\n"
            "Patient: About 2 out of 10. It is quite manageable.\n"
            "Agent: Did you attend your follow-up appointment last week?\n"
            "Patient: Yes, I went on Tuesday. The doctor was very happy with "
            "my progress.\n"
            "Agent: And do you have any emergency concerns to share?\n"
            "Patient: No, nothing at all. Everything feels fine.\n"
            "Agent: Wonderful. We will call again next week. Take care.\n"
            "Patient: Thank you very much. Goodbye."
        ),
        "expected_action": "accept",
        "mock_llm_evidence": [
            {
                "field_name": "symptom_update",
                "extracted_value": "incision healing well; no new symptoms",
                "citations": ["The incision is healing well. I have no new symptoms."],
                "confidence": 0.94,
                "reasoning": "Patient explicitly described improvement.",
            },
            {
                "field_name": "medication_adherence",
                "extracted_value": "full adherence",
                "citations": ["I take all my tablets every morning without missing a dose."],
                "confidence": 0.97,
                "reasoning": "Explicit daily adherence confirmed.",
            },
            {
                "field_name": "pain_level",
                "extracted_value": "2/10",
                "citations": ["About 2 out of 10. It is quite manageable."],
                "confidence": 0.96,
                "reasoning": "Numeric pain rating given directly.",
            },
            {
                "field_name": "appointment_compliance",
                "extracted_value": "attended (Tuesday)",
                "citations": ["Yes, I went on Tuesday. The doctor was very happy with my progress."],
                "confidence": 0.95,
                "reasoning": "Appointment confirmed with day and outcome.",
            },
            {
                "field_name": "emergency_concerns",
                "extracted_value": None,
                "citations": [],
                "confidence": 0.00,
                "reasoning": "Patient explicitly denied emergency concerns.",
            },
        ],
        "mock_raw_llm_output": {},
        "summary_for_rag": (
            "Week 4 post-op. Incision healing well, no new symptoms. "
            "Pain 2/10. Full medication adherence. Tuesday appointment attended; "
            "doctor satisfied with progress. No emergency concerns."
        ),
        "rag_topics": ["wound_healing", "medication", "pain"],
    },

    # -- CALL-DEMO-B: behavioral farewell indicators --------------------
    {
        "scenario_id": "CALL-DEMO-B",
        "category": "risk_indirect",
        "description": "Behavioral farewell -- indirect suicidal ideation",
        "transcript": (
            "Agent: Good afternoon. This is your weekly follow-up call. "
            "How have you been keeping?\n"
            "Patient: I have been sorting through my belongings. I have been "
            "giving away some of my possessions -- things I have had for years. "
            "My daughter asked why and I did not really have an answer.\n"
            "Agent: That sounds like a significant change. Can you tell me more "
            "about how you have been feeling?\n"
            "Patient: I just feel like I want to say the things I never said. "
            "I have been writing letters to my grandchildren, just to share "
            "things with them before it is too late.\n"
            "Agent: I want to make sure you are okay. Are you having any thoughts "
            "of harming yourself?\n"
            "Patient: Not exactly. I just feel like everything is winding down. "
            "Pain is about 5 out of 10. I have been taking my medication.\n"
            "Agent: Thank you for being honest with me. I am going to make sure "
            "your care team hears about this.\n"
            "Patient: Okay. Thank you."
        ),
        "expected_action": "triage_risk",
        "mock_llm_evidence": [
            {
                "field_name": "symptom_update",
                "extracted_value": "emotional distress; behavioral farewell indicators",
                "citations": ["I just feel like everything is winding down."],
                "confidence": 0.82,
                "reasoning": "Patient described giving away possessions and writing "
                             "farewell letters -- indirect suicidal ideation signals.",
            },
            {
                "field_name": "medication_adherence",
                "extracted_value": "taking medication",
                "citations": ["I have been taking my medication."],
                "confidence": 0.88,
                "reasoning": "Patient confirmed taking medication without specifics.",
            },
            {
                "field_name": "pain_level",
                "extracted_value": "5/10",
                "citations": ["Pain is about 5 out of 10."],
                "confidence": 0.91,
                "reasoning": "Numeric rating given.",
            },
            {
                "field_name": "appointment_compliance",
                "extracted_value": None,
                "citations": [],
                "confidence": 0.00,
                "reasoning": "Appointments not discussed during this call.",
            },
            {
                "field_name": "emergency_concerns",
                "extracted_value": "indirect suicidal ideation: giving away possessions, "
                                   "writing farewell letters, sense of finality",
                "citations": [
                    "giving away some of my possessions",
                    "writing letters to my grandchildren, just to share things "
                    "with them before it is too late",
                ],
                "confidence": 0.88,
                "reasoning": "Classic behavioral farewell indicators with explicit "
                             "'before it is too late' language.",
            },
        ],
        "mock_raw_llm_output": {},
        "summary_for_rag": (
            "Patient reported giving away possessions and writing farewell letters "
            "to grandchildren 'before it is too late'. Denied direct self-harm intent "
            "but described feeling 'everything is winding down'. Pain 5/10. Medication "
            "taken. Emergency escalation triggered."
        ),
        "rag_topics": ["risk_indirect", "behavioral_farewell", "medication"],
    },

    # -- CALL-DEMO-C: degraded transcription ---------------------------
    {
        "scenario_id": "CALL-DEMO-C",
        "category": "transcription_bad",
        "description": "High ASR failure rate -- routed for human review",
        "transcript": (
            "Agent: Good morning. How have you been feeling this week?\n"
            "Patient: I have been feeling [unintelligible] since the last time "
            "we spoke. [unintelligible] is a bit [unintelligible].\n"
            "Agent: I see. Are you taking your medications as prescribed?\n"
            "Patient: [unintelligible] every day I think. Sometimes [unintelligible] "
            "after breakfast, [unintelligible].\n"
            "Agent: On a scale of 1 to 10, what is your pain level?\n"
            "Patient: About [unintelligible] out of 10, maybe [unintelligible].\n"
            "Agent: Did you attend your appointment?\n"
            "Patient: [unintelligible] I went to [unintelligible] on Thursday "
            "[unintelligible].\n"
            "Agent: Any emergency concerns?\n"
            "Patient: No, nothing urgent [unintelligible] fine.\n"
            "Agent: I am sorry the line quality is poor today. We will arrange "
            "a follow-up.\n"
            "Patient: Thank you [unintelligible] goodbye."
        ),
        "expected_action": "triage_ambiguous",
        "mock_llm_evidence": [
            {
                "field_name": "symptom_update",
                "extracted_value": "unclear -- significant transcription gaps",
                "citations": [],
                "confidence": 0.25,
                "reasoning": "Multiple unintelligible segments make this unreliable.",
            },
            {
                "field_name": "medication_adherence",
                "extracted_value": "possibly taking daily",
                "citations": ["every day I think"],
                "confidence": 0.32,
                "reasoning": "Tentative confirmation only; gaps obscure details.",
            },
            {
                "field_name": "pain_level",
                "extracted_value": "unknown",
                "citations": [],
                "confidence": 0.20,
                "reasoning": "Numeric portion was unintelligible.",
            },
            {
                "field_name": "appointment_compliance",
                "extracted_value": "possibly attended Thursday",
                "citations": ["I went to [unintelligible] on Thursday"],
                "confidence": 0.38,
                "reasoning": "Partial confirmation; destination not heard.",
            },
            {
                "field_name": "emergency_concerns",
                "extracted_value": None,
                "citations": [],
                "confidence": 0.00,
                "reasoning": "Patient appeared to deny concerns; cannot be certain.",
            },
        ],
        "mock_raw_llm_output": {},
        "summary_for_rag": (
            "Call quality severely degraded. Multiple unintelligible segments "
            "across all required fields. Cannot confirm medication adherence, "
            "pain level, or appointment. Routed to human review for callback."
        ),
        "rag_topics": ["transcription_quality", "callback_required"],
    },
]

# ---------------------------------------------------------------------------
# Scenario loader (supports demo calls + eval JSONL scenarios)
# ---------------------------------------------------------------------------

def _load_scenario(scenario_id: str | None, jsonl_path: Path) -> dict:
    """Return a demo call dict by ID, checking demo list first then JSONL."""
    if scenario_id is None:
        return None  # caller will iterate all DEMO_CALLS

    # Check demo calls
    for sc in DEMO_CALLS:
        if sc["scenario_id"] == scenario_id:
            return sc

    # Fall through to eval JSONL
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sc = json.loads(line)
                if sc.get("scenario_id") == scenario_id:
                    # Normalise: eval JSONL scenarios don't have summary_for_rag/rag_topics
                    sc.setdefault("summary_for_rag", sc.get("description", ""))
                    sc.setdefault("rag_topics", [sc.get("category", "")])
                    return sc

    raise SystemExit(f"Scenario {scenario_id!r} not found in demo calls or {jsonl_path}")


# ---------------------------------------------------------------------------
# Extractor factories
# ---------------------------------------------------------------------------

def _make_extractor(mode: str):
    if mode == "real":
        from eval.real_extractor import RealExtractor
        return RealExtractor()
    from eval.extractor import MockExtractor
    return MockExtractor()


def _try_make_planner(rag: CallHistoryRAG, profile_store: ProfileStore) -> object:
    """
    Return a real CallPlanner if OPENAI_API_KEY is available, otherwise
    return a sentinel object that always returns the deterministic fallback.
    """
    try:
        from agents.planner import CallPlanner
        return CallPlanner(rag=rag, profile_store=profile_store,
                           clinic_name="City Medical Centre")
    except EnvironmentError:
        return None


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

DEFAULT_CALL_SIGNALS = CallSignals(
    avg_response_latency_ms=1200.0,
    pause_frequency=1.1,
    interruption_count=0,
    call_duration_s=195.0,
    speech_rate_wpm=112.0,
)


def _run_stage0_preplan(
    patient_id: str,
    planner,
    profile_store: ProfileStore,
    rag: CallHistoryRAG,
) -> None:
    t0 = time.perf_counter()
    profile = profile_store.get_profile(patient_id)
    history = rag.retrieve_recent(patient_id, n=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _stage(0, "Pre-call preparation", elapsed_ms)
    _kv("Patient", patient_id)
    _kv("Call history", f"{rag.count(patient_id)} call(s) in memory")
    _kv("Profile confidence",
        f"{profile.confidence:.2f} ({profile.call_count} calls)  "
        f"{'personalised' if not profile.use_default_script else 'neutral script'}")

    if history:
        print("      Recent history:")
        for r in history[:2]:
            print(f"        {r.call_id}  {r.timestamp[:10]}  "
                  f"topics: {', '.join(r.topics or ['-'])}")

    if planner is None:
        print("      [CallPlanner] OPENAI_API_KEY not set -- using deterministic fallback plan")
        _show_fallback_plan()
        return

    t1 = time.perf_counter()
    plan = planner.plan_call(patient_id)
    plan_ms = (time.perf_counter() - t1) * 1000

    mode = "personalised" if plan.personalization_used else "neutral (default)"
    fallback_tag = "  [FALLBACK]" if plan.fallback_used else ""
    print(f"      [CallPlanner] plan generated in {plan_ms:.0f} ms  "
          f"({mode}){fallback_tag}")
    _kv("Opening", f'"{plan.opening[:70]}..."' if len(plan.opening) > 70 else f'"{plan.opening}"')
    print("      Questions:")
    for q in plan.questions[:5]:
        flag = " [history]" if q.from_history else ""
        print(f"        P{q.priority}  {q.field_name:<24} {q.question[:54]}{flag}")
    if plan.notes:
        for note in plan.notes[:2]:
            print(f"      NOTE  {note}")


def _show_fallback_plan() -> None:
    from agents.models import DEFAULT_QUESTIONS
    print("      [CallPlanner] Fallback plan questions:")
    for fn, q in list(DEFAULT_QUESTIONS.items())[:3]:
        print(f"        P2  {fn:<24} {q[:54]}")
    print(f"        ... (+ {len(DEFAULT_QUESTIONS) - 3} more)")


def _run_stage1_transcript(scenario: dict) -> None:
    _stage(1, "Transcript received from CALL-E")
    _kv("Call ID", scenario["scenario_id"])
    _kv("Category", scenario["category"])
    _kv("Description", scenario.get("description", ""))
    words = len(scenario["transcript"].split())
    unintelligible = scenario["transcript"].count("[unintelligible]")
    _kv("Transcript", f"{words} words  |  {unintelligible} [unintelligible] markers")
    print()
    # Print transcript with indent, capped at 12 lines
    lines = scenario["transcript"].splitlines()
    for line in lines[:12]:
        print(f"      {line}")
    if len(lines) > 12:
        print(f"      ... ({len(lines) - 12} more lines)")


def _run_stage2_extraction(
    scenario: dict,
    extractor,
) -> tuple[list, dict, float]:
    t0 = time.perf_counter()
    llm_evidence, raw_output = extractor.extract(scenario["transcript"], scenario)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _stage(2, "LLM field extraction", elapsed_ms)
    model = raw_output.get("_model", "mock")
    prompt_tok = raw_output.get("_prompt_tokens", "-")
    compl_tok  = raw_output.get("_completion_tokens", "-")
    _kv("Model", model)
    if prompt_tok != "-":
        _kv("Tokens", f"prompt {prompt_tok}  completion {compl_tok}")

    print()
    print(f"      {'FIELD':<26} {'VALUE':<28} CONF")
    print(f"      {'-'*26} {'-'*28} {'-'*4}")
    for ev in llm_evidence:
        val = str(ev.extracted_value or "(none)")[:27]
        conf_bar = "#" * int(ev.confidence * 10)
        print(f"      {ev.field_name:<26} {val:<28} {ev.confidence:.2f}  {conf_bar}")

    return llm_evidence, raw_output, elapsed_ms


def _run_stage3_gate(
    scenario: dict,
    llm_evidence: list,
    raw_output: dict,
    gate: ArbitrationGate,
) -> tuple:
    inp = ArbitrationInput(
        call_id=scenario["scenario_id"],
        patient_id="PAT-DEMO",
        transcript=scenario["transcript"],
        llm_evidence=llm_evidence,
        call_signals=DEFAULT_CALL_SIGNALS,
        raw_llm_output=raw_output,
    )
    t0 = time.perf_counter()
    result = gate.evaluate(inp)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _stage(3, "Arbitration gate -- 7 dimensions", elapsed_ms)
    print(f"\n      {'DIMENSION':<26} {'STATUS':<6} {'SCORE':>6}  FLAGS / REASON")
    print(f"      {'-'*26} {'-'*6} {'-'*6}  {'-'*28}")

    for dim_enum in DIMENSION_ORDER:
        key = dim_enum.value
        if key not in result.dimensions:
            continue
        dim = result.dimensions[key]
        label = DIMENSION_LABELS.get(dim_enum, key)
        status = "PASS" if dim.passed else "FAIL"
        flags  = "  ".join(dim.flags[:2]) if dim.flags else ""
        if len(dim.flags) > 2:
            flags += f"  (+{len(dim.flags)-2})"
        print(f"      {label:<26} {status:<6} {dim.score:>5.2f}  {flags[:46]}")

    return result, elapsed_ms


def _run_stage4_decision(result) -> None:
    _stage(4, "Triage decision")

    action = result.action
    _kv("Action", action.value.upper())

    if result.risk_flags:
        _kv("Risk flags", "  ".join(result.risk_flags))

    print()

    if action == Action.ACCEPT:
        print("      " + "-" * 62)
        print("      CALL ACCEPTED -- all dimensions passed.")
        print("      No human review required.")
        print("      " + "-" * 62)

    elif action == Action.TRIAGE_RISK:
        print("      " + "!" * 62)
        print("      *** URGENT ESCALATION / \u7d27\u6025\u98ce\u9669\u5347\u7ea7 ***")
        print()
        print("      Risk signals detected in this call.")
        print("      Routing to the on-call clinical team immediately.")
        print("      Priority 1 -- action required within 15 minutes.")
        if result.risk_flags:
            print()
            print("      Detected flags:")
            for f in result.risk_flags:
                print(f"        -- {f}")
        print("      " + "!" * 62)

    elif action == Action.TRIAGE_AMBIGUOUS:
        print("      " + "~" * 62)
        print("      !!  \u5df2\u6807\u8bb0\u9700\u4eba\u5de5\u5ba1\u6838")
        print("      !!  Flagged for human review")
        print()
        print("      One or more quality dimensions failed.")
        print("      Routing to the human review queue (Priority 2).")
        failing = [
            DIMENSION_LABELS.get(d, d.value)
            for d in DIMENSION_ORDER
            if d.value in result.dimensions and not result.dimensions[d.value].passed
        ]
        if failing:
            print(f"      Failing dimensions: {', '.join(failing)}")
        print("      " + "~" * 62)

    elif action == Action.REJECT:
        print("      " + "-" * 62)
        print("      CALL REJECTED -- out-of-domain or PII violation.")
        print("      Not routed for clinical review.")
        failing = [
            DIMENSION_LABELS.get(d, d.value)
            for d in DIMENSION_ORDER
            if d.value in result.dimensions and not result.dimensions[d.value].passed
        ]
        if failing:
            print(f"      Failing dimensions: {', '.join(failing)}")
        print("      " + "-" * 62)


def _run_stage5_memory(
    patient_id: str,
    scenario: dict,
    result,
    profile_store: ProfileStore,
    rag: CallHistoryRAG,
) -> None:
    _stage(5, "Post-call memory update")

    # Profile update
    t0 = time.perf_counter()
    profile_store.update(patient_id, DEFAULT_CALL_SIGNALS)
    profile = profile_store.get_profile(patient_id)
    profile_ms = (time.perf_counter() - t0) * 1000

    _kv("ProfileStore",
        f"call_count={profile.call_count}  "
        f"confidence={profile.confidence:.2f}  "
        f"({profile_ms:.1f} ms)")

    # Agents: confirmation check on low-confidence fields
    low_conf = [
        ev for ev in (result._inp_evidence if hasattr(result, "_inp_evidence") else [])
        if ev.confidence < 0.60
    ]
    # We don't have inp on result directly; let's pull from the extraction stage via closure
    # Instead just note it structurally
    print("      [ConfirmationTracker] would escalate fields with conf < 0.60 in live call")

    # RAG update
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = scenario.get("summary_for_rag", scenario.get("description", ""))
    topics  = scenario.get("rag_topics", [scenario.get("category", "")])

    t0 = time.perf_counter()
    rag.add_call_summary(
        patient_id=patient_id,
        call_id=scenario["scenario_id"],
        summary=summary,
        timestamp=ts,
        topics=topics,
    )
    rag_ms = (time.perf_counter() - t0) * 1000
    _kv("CallHistoryRAG",
        f"summary added  ({rag_ms:.1f} ms)  "
        f"total for patient: {rag.count(patient_id)}")


def _run_stage5_confirmation_probe(llm_evidence: list) -> None:
    """Show which fields would trigger confirmation / symptom probing in a live call."""
    print()
    print("      [agents/ in-call layer -- hypothetical for this call]")

    tracker = ConfirmationTracker()
    conf_needed = []
    for ev in llm_evidence:
        if ev.extracted_value is not None:
            decision = tracker.decide(
                field_name=ev.field_name,
                confidence=ev.confidence,
                candidates=[{"text": str(ev.extracted_value), "confidence": ev.confidence}],
            )
            if decision.action.value != "accept":
                conf_needed.append((ev.field_name, decision.level.name, decision.utterance))

    if conf_needed:
        print("      Confirmation needed:")
        for fn, level, utterance in conf_needed:
            u_short = utterance[:54] if utterance else "(route to human)"
            print(f"        {fn:<26} {level:<8}  \"{u_short}\"")
    else:
        print("      All fields above confirmation threshold -- no re-ask needed")

    # Symptom probe -- classify_ambiguity takes a duck-typed claim object
    class _ClaimProxy:
        def __init__(self, _ev):
            self.field_name      = _ev.field_name
            self.confidence      = _ev.confidence
            self.extracted_value = _ev.extracted_value

    for ev in llm_evidence:
        if ev.field_name in ("symptom_update", "pain_level") and ev.extracted_value:
            if classify_ambiguity(_ClaimProxy(ev)) == AmbiguityKind.UNDERSPECIFIED:
                if is_symptom_incomplete(str(ev.extracted_value)):
                    probe = build_probe_plan(ev.field_name, str(ev.extracted_value))
                    if probe.has_questions:
                        print(f"      Symptom probe ({ev.field_name}):")
                        for q in probe.questions[:2]:
                            print(f"        -> {q}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_one_call(
    scenario: dict,
    extractor,
    planner,
    profile_store: ProfileStore,
    rag: CallHistoryRAG,
    gate: ArbitrationGate,
) -> None:
    patient_id = "PAT-DEMO"
    call_t0 = time.perf_counter()

    _header(
        f"CALL  {scenario['scenario_id']}  |  {scenario['description']}"
    )

    # -- Stage 0: Pre-call ---------------------------------------------
    _run_stage0_preplan(patient_id, planner, profile_store, rag)

    # -- Stage 1: Transcript -------------------------------------------
    _run_stage1_transcript(scenario)

    # -- Stage 2: Extraction -------------------------------------------
    llm_evidence, raw_output, _ext_ms = _run_stage2_extraction(scenario, extractor)

    # -- Stage 3: Gate -------------------------------------------------
    result, _gate_ms = _run_stage3_gate(scenario, llm_evidence, raw_output, gate)

    # -- Stage 4: Decision ---------------------------------------------
    _run_stage4_decision(result)

    # -- Stage 5: Memory -----------------------------------------------
    _run_stage5_memory(patient_id, scenario, result, profile_store, rag)

    # -- In-call agent layer -------------------------------------------
    _run_stage5_confirmation_probe(llm_evidence)

    total_ms = (time.perf_counter() - call_t0) * 1000
    expected = scenario.get("expected_action", "?")
    actual   = result.action.value
    verdict  = "PASS" if actual == expected else "FAIL (expected: " + expected + ")"
    print()
    _rule("=")
    print(f"  TOTAL TIME: {total_ms:.1f} ms  |  RESULT: {actual.upper()}  |  {verdict}")
    _rule("=")


def main() -> None:
    # Reconfigure stdout to UTF-8 so Chinese characters survive on Windows
    # (Python defaults to the system code page, often GBK on Chinese Windows).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="FollowUp Companion end-to-end pipeline demo."
    )
    parser.add_argument(
        "--extractor", choices=["mock", "real"], default="real",
        help="LLM extractor (default: real; needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--scenario", default=None, metavar="ID",
        help="Run a single scenario by ID (e.g. SC-004, CALL-DEMO-B). "
             "Default: run all three demo calls.",
    )
    args = parser.parse_args()

    jsonl_path = ROOT / "eval" / "test_scenarios.jsonl"

    # -- Bootstrap shared resources ------------------------------------
    profile_store = ProfileStore()          # in-memory; no persistence for demo
    rag           = CallHistoryRAG(persist_dir=None, collection_name="demo_run")
    gate          = ArbitrationGate()
    extractor     = _make_extractor(args.extractor)
    planner       = _try_make_planner(rag, profile_store)

    _header("FOLLOWUP COMPANION -- END-TO-END PIPELINE DEMO", char="#")
    print(f"  Extractor   : {args.extractor}")
    print(f"  CallPlanner : {'real (gpt-4o-mini)' if planner else 'deterministic fallback (no key)'}")
    print(f"  Memory      : in-memory (ephemeral for this demo)")

    # -- Select scenarios ----------------------------------------------
    if args.scenario:
        sc = _load_scenario(args.scenario, jsonl_path)
        scenarios = [sc]
    else:
        scenarios = DEMO_CALLS

    # -- Run -----------------------------------------------------------
    for sc in scenarios:
        run_one_call(sc, extractor, planner, profile_store, rag, gate)
        print()

    # -- Token usage summary -------------------------------------------
    usage = extractor.token_usage
    if usage:
        print()
        _rule("-")
        print("  TOKEN USAGE (extractor)")
        _kv("Model", usage["model"])
        _kv("API calls", str(usage["calls"]))
        _kv("Total tokens",
            f"{usage['total_tokens']:,}  "
            f"(prompt {usage['prompt_tokens']:,}  "
            f"completion {usage['completion_tokens']:,})")
        _kv("Estimated cost", f"${usage['estimated_cost_usd']:.4f} USD")
        if planner and hasattr(planner, "token_usage"):
            pu = planner.token_usage
            if pu.get("calls", 0):
                print()
                print("  TOKEN USAGE (planner)")
                _kv("API calls", str(pu["calls"]))
                _kv("Total tokens", f"{pu['total_tokens']:,}")
                _kv("Estimated cost", f"${pu['estimated_cost_usd']:.4f} USD")
        _rule("-")


if __name__ == "__main__":
    main()
