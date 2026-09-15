"""
CallPlanner - deciding what to ask before the phone rings.

The gap this fills
------------------
eval/extractor.py understands a call after the fact. Nothing decided what to
*say*. This module produces the pre-call script: the opening, the specific
questions, and the pacing to use.

Three inputs
------------
  memory/call_history_rag.py   what previous calls were about
  memory/profile_store.py      how this patient communicates, with a
                               confidence score attached
  STANDARD_PROTOCOL_ITEMS      what the follow-up protocol requires

Who decides what
----------------
The model writes wording. Rules decide everything else - the same split as
arbitration/, for the same reason (auditability):

  * Pacing directives are computed from the profile in code
    (derive_tone_directives), then handed to the model as constraints. The
    model never reads raw latency numbers and invents its own pacing.
  * Personalisation is gated on profile.use_default_script, not on the
    model's judgement. Below PERSONALIZATION_CONFIDENCE_THRESHOLD the plan
    uses neutral wording and drops conversational callbacks to previous
    calls, because a profile built from one or two calls is not evidence.
    Clinical facts from history are still used to target questions - those
    are recorded events, not inferred traits.
  * Every mandatory protocol item is verified present after generation, and
    added back from a fixed template if the model dropped it.

Degradation
-----------
If the API call fails or returns unparseable JSON, plan_call() returns a
deterministic plan built from the protocol templates with
`fallback_used=True` and the reason in `notes`. A follow-up call with a
plain script beats no call, which is the same posture as providers/.

Model
-----
gpt-4o-mini, matching eval/real_extractor.py. Planning is a short
structured-writing task; a frontier model is not worth the cost here.
Temperature 0.3 rather than the extractor's 0.0 - phrasing benefits from a
little variation, and nothing downstream depends on it being reproducible.

Usage
-----
    from memory.call_history_rag import CallHistoryRAG
    from memory.profile_store import ProfileStore

    planner = CallPlanner(rag=CallHistoryRAG(), profile_store=ProfileStore(...))
    plan = planner.plan_call("PAT-001")

    say(plan.opening)
    for q in plan.questions:
        say(q.question)
    say(plan.closing)

    planner.print_token_usage()
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional, Sequence

from agents.confirmation import ConfirmationTracker, decide_confirmation
from agents.models import (
    STANDARD_PROTOCOL_ITEMS,
    CallPlan,
    PlannedQuestion,
    ProtocolItem,
    SymptomProbePlan,
)
from agents.symptom_probe import build_probe_plan, classify_ambiguity, plan_for_claim
from memory.profile_store import (
    PERSONALIZATION_CONFIDENCE_THRESHOLD,
    PersonalizationProfile,
)

# Load .env before reading env vars (same pattern as eval/real_extractor.py).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"

# Higher than the extractor's 0.0: this is phrasing, not data extraction.
TEMPERATURE = 0.3

# USD per 1M tokens (prompt, completion), as of 2025-09.
PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# How many historical summaries to retrieve by default.
DEFAULT_HISTORY_COUNT = 3

# Summaries are truncated before going into the prompt; the opening lines
# carry the clinical substance and this keeps the prompt cost predictable.
HISTORY_SUMMARY_CHAR_LIMIT = 400


# ---------------------------------------------------------------------------
# Pacing thresholds
# ---------------------------------------------------------------------------
# Derived from the objective signals in memory/profile_store.py. Only applied
# when the profile is trustworthy (use_default_script is False).

SLOW_RESPONSE_LATENCY_MS = 2500.0   # Patient needs time before answering.
HIGH_PAUSE_FREQUENCY = 3.0          # Pauses per minute; suggests effortful speech.
HIGH_INTERRUPTION_COUNT = 2.0       # Patient tends to cut in - keep it short.
SLOW_SPEECH_RATE_WPM = 100.0        # Slow speech; mirror it and simplify.
SHORT_CALL_DURATION_S = 120.0       # Short calls historically - front-load.

NEUTRAL_TONE_DIRECTIVES: tuple[str, ...] = (
    "Use a neutral, standard greeting and wording. Do not personalise.",
    "Moderate pace. Ask one question at a time and wait for the answer.",
    "Do not bring up personal topics or life details from previous calls.",
)


# ---------------------------------------------------------------------------
# Fixed templates
# ---------------------------------------------------------------------------
# Used for the degraded plan and to restore any mandatory item the model
# dropped. Plain, neutral, and safe to speak to any patient.

DEFAULT_QUESTIONS: dict[str, str] = {
    "symptom_update": (
        "Since we last spoke, have you noticed any new symptoms, or "
        "anything that feels different?"
    ),
    "medication_adherence": (
        "Are you still taking all your medication the way your doctor "
        "prescribed it?"
    ),
    "pain_level": (
        "On a scale where 0 is no pain and 10 is the worst you can imagine, "
        "where would you put it right now?"
    ),
    "appointment_compliance": (
        "Did you manage to get to your last scheduled appointment?"
    ),
    "emergency_concerns": (
        "Has anything come up that worried you, or that felt like it needed "
        "attention straight away?"
    ),
}

DEFAULT_OPENING = (
    "Hello, this is the follow-up team at {clinic}. I'd like to check in on "
    "how you've been getting on since we last spoke - it should only take a "
    "few minutes. Is now a good time?"
)

DEFAULT_CLOSING = (
    "That's everything I needed to ask. If anything changes before we speak "
    "again, please do get in touch with us. Take care, goodbye."
)

DEFAULT_CLINIC_NAME = "the clinic"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a call planner for a medical telephone follow-up \
system serving chronic and post-operative patients. You write the script the \
agent will speak, before the call is placed.

TASK
Return a JSON object:
{
  "opening": "<what the agent says first>",
  "questions": [
    {
      "field_name": "<protocol field name, or \\"followup\\" for a history-driven question>",
      "question": "<the question as it will be spoken>",
      "rationale": "<one short sentence, English, why this wording>",
      "priority": <1 = ask first, 2 = normal, 3 = only if time allows>
    }
  ],
  "continuity_topics": ["<topic from a previous call you picked up on>"],
  "closing": "<how the agent ends the call>"
}

HARD CONSTRAINTS
  1. Every field_name listed under MANDATORY PROTOCOL ITEMS must appear \
exactly once in "questions". You may reword and reorder them; you may not \
drop or merge them.
  2. All patient-facing text (opening, question, closing) must be in plain \
conversational English, suitable for reading aloud to an elderly patient on \
the phone. Keep each question to one sentence and avoid medical jargon.
  3. Follow every directive under TONE DIRECTIVES. They are derived from \
this patient's measured call behaviour and are not suggestions.
  4. When PERSONALIZATION is "disabled", the opening must be a neutral \
standard greeting and "continuity_topics" must be an empty list. Do not \
reference anything personal from previous calls. Clinical facts may still \
inform which questions you ask.
  5. Questions listed under CARRY-OVER QUESTIONS must be included verbatim \
with priority 1.

SAFETY CONSTRAINTS
  * Ask; never advise. No diagnosis, no treatment or medication \
recommendations, no interpretation of test results, no reassurance about \
whether a symptom is serious.
  * Never state or imply a clinical conclusion in a question.
  * Do not read back personal identifiers (ID numbers, addresses, phone \
numbers) to the patient.
  * One question per entry. Do not stack two questions in one sentence - \
patients answer only the last one.

Return ONLY the JSON object. No markdown, no commentary."""


# ---------------------------------------------------------------------------
# Deterministic pacing rules
# ---------------------------------------------------------------------------

def derive_tone_directives(profile: PersonalizationProfile) -> list[str]:
    """
    Turn measured call signals into pacing instructions.

    Deterministic and inspectable: the model receives the conclusions, not
    the raw numbers, so it cannot improvise its own pacing policy from a
    latency figure.

    Returns NEUTRAL_TONE_DIRECTIVES unchanged when the profile is not
    trustworthy - see memory/profile_store.py for the confidence ramp.
    """
    if profile.use_default_script:
        return list(NEUTRAL_TONE_DIRECTIVES)

    directives: list[str] = []

    if (profile.avg_response_latency_ms or 0) >= SLOW_RESPONSE_LATENCY_MS:
        directives.append(
            "This patient needs time before answering. Leave a clear pause "
            "after each question. Do not rush them or re-ask straight away."
        )
    if (profile.avg_pause_frequency or 0) >= HIGH_PAUSE_FREQUENCY:
        directives.append(
            "This patient pauses often while speaking. Use short sentences "
            "and ask about one thing at a time; avoid compound questions."
        )
    if (profile.avg_interruption_count or 0) >= HIGH_INTERRUPTION_COUNT:
        directives.append(
            "This patient tends to speak over the agent. Keep questions "
            "brief and let them finish before moving on."
        )
    if profile.avg_speech_rate_wpm is not None and (
        profile.avg_speech_rate_wpm <= SLOW_SPEECH_RATE_WPM
    ):
        directives.append(
            "This patient speaks slowly. Slow your own pace, use everyday "
            "words, and avoid medical terminology."
        )
    if (profile.avg_call_duration_s or 0) <= SHORT_CALL_DURATION_S:
        directives.append(
            "This patient's calls are usually short. Cover the mandatory "
            "items first and put the most important questions up front."
        )

    if not directives:
        # Profile is trustworthy and shows no special needs.
        directives.append(
            "This patient communicates comfortably. Keep a natural, concise "
            "pace and ask one question at a time."
        )
    return directives


def _history_topics(history: Sequence[Any]) -> list[str]:
    """Distinct topic tags across the retrieved summaries, order preserved."""
    topics: list[str] = []
    for item in history:
        for topic in getattr(item, "topics", []) or []:
            if topic not in topics:
                topics.append(topic)
    return topics


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _format_profile_block(profile: PersonalizationProfile) -> str:
    """
    Describe the profile for the prompt.

    When personalisation is disabled the averaged signals are withheld
    entirely rather than shown with a caveat - a model given numbers tends
    to use them, and at this confidence they are not evidence.
    """
    lines = [
        f"call_count: {profile.call_count}",
        f"profile_confidence: {profile.confidence:.2f} "
        f"(threshold {PERSONALIZATION_CONFIDENCE_THRESHOLD:.2f})",
    ]
    if profile.use_default_script:
        lines.append(
            "PERSONALIZATION: disabled - too few recorded calls for the "
            "profile to be reliable. Use neutral defaults."
        )
        return "\n".join(lines)

    lines.append("PERSONALIZATION: enabled")
    if profile.avg_response_latency_ms is not None:
        lines.append(f"avg_response_latency_ms: {profile.avg_response_latency_ms:.0f}")
    if profile.avg_pause_frequency is not None:
        lines.append(f"avg_pause_frequency: {profile.avg_pause_frequency:.2f}")
    if profile.avg_speech_rate_wpm is not None:
        lines.append(f"avg_speech_rate_wpm: {profile.avg_speech_rate_wpm:.0f}")
    if profile.avg_interruption_count is not None:
        lines.append(f"avg_interruption_count: {profile.avg_interruption_count:.2f}")
    if profile.avg_call_duration_s is not None:
        lines.append(f"avg_call_duration_s: {profile.avg_call_duration_s:.0f}")
    return "\n".join(lines)


def _format_history_block(
    history: Sequence[Any],
    personalization_enabled: bool,
) -> str:
    if not history:
        return "(no previous calls on record - this is a first contact)"

    lines: list[str] = []
    for item in history:
        summary = str(getattr(item, "summary", ""))[:HISTORY_SUMMARY_CHAR_LIMIT]
        lines.append(
            f"- call_id: {getattr(item, 'call_id', '?')} "
            f"| date: {getattr(item, 'timestamp', '?')} "
            f"| topics: {', '.join(getattr(item, 'topics', []) or []) or '-'}\n"
            f"  summary: {summary}"
        )
    block = "\n".join(lines)
    if not personalization_enabled:
        block += (
            "\n\nNOTE: use these only to decide which clinical questions to "
            "ask. Do not refer to them conversationally and do not mention "
            "personal details from them."
        )
    return block


def _format_protocol_block(protocol: Sequence[ProtocolItem]) -> str:
    return "\n".join(
        f"- {item.field_name} ({item.label})"
        f"{'' if item.mandatory else ' [optional]'}: {item.intent}"
        for item in protocol
    )


def _build_user_prompt(
    patient_id: str,
    profile: PersonalizationProfile,
    history: Sequence[Any],
    protocol: Sequence[ProtocolItem],
    carry_over: Sequence[str],
    tone_directives: Sequence[str],
    clinic_name: str,
    extra_context: str = "",
) -> str:
    personalization_enabled = not profile.use_default_script
    sections = [
        f"PATIENT: {patient_id}",
        f"CLINIC NAME (use in the opening): {clinic_name}",
        "",
        "COMMUNICATION PROFILE",
        _format_profile_block(profile),
        "",
        "TONE DIRECTIVES (mandatory)",
        "\n".join(f"- {d}" for d in tone_directives),
        "",
        "PREVIOUS CALLS (most relevant first)",
        _format_history_block(history, personalization_enabled),
        "",
        "MANDATORY PROTOCOL ITEMS (each must appear exactly once)",
        _format_protocol_block(protocol),
    ]
    if carry_over:
        sections += [
            "",
            "CARRY-OVER QUESTIONS (include verbatim, priority 1)",
            "\n".join(f"- {q}" for q in carry_over),
        ]
    if extra_context:
        sections += ["", "ADDITIONAL CONTEXT", extra_context]
    sections += ["", "Produce the call plan as JSON."]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CallPlanner
# ---------------------------------------------------------------------------

class CallPlanner:
    """
    Builds a pre-call script from history, profile and protocol.

    Parameters
    ----------
    api_key : str | None
        OpenAI key; falls back to OPENAI_API_KEY. Not required when
        `client` is injected.
    rag :
        A memory.call_history_rag.CallHistoryRAG (or anything exposing
        `retrieve` / `retrieve_recent`). None means plan without history.
        Injected rather than constructed here so that importing this module
        does not pull in chromadb.
    profile_store :
        A memory.profile_store.ProfileStore, or None to plan with a
        zero-confidence profile (neutral script).
    client :
        Pre-built OpenAI client. Mainly for tests; when given, no API key
        is needed and the openai package is not imported.
    model : str
        Chat model id. Cost is reported for entries in PRICING_USD_PER_1M.
    clinic_name : str
        Name spoken in the opening.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rag: Any = None,
        profile_store: Any = None,
        client: Any = None,
        model: str = MODEL,
        clinic_name: str = DEFAULT_CLINIC_NAME,
    ) -> None:
        self._rag = rag
        self._profile_store = profile_store
        self._model = model
        self._clinic_name = clinic_name

        if client is not None:
            self._client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file or set it in the shell environment."
                )
            # Imported lazily so agents/ stays importable (and testable)
            # without the openai package installed.
            try:
                from openai import OpenAI
            except ImportError:
                raise SystemExit(
                    "openai package is not installed. Run: pip install openai"
                )
            self._client = OpenAI(api_key=key)

        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_count = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan_call(
        self,
        patient_id: str,
        protocol_items: Sequence[ProtocolItem] = STANDARD_PROTOCOL_ITEMS,
        retrieval_query: Optional[str] = None,
        n_history: int = DEFAULT_HISTORY_COUNT,
        carry_over_symptoms: Optional[Iterable[str]] = None,
        extra_context: str = "",
    ) -> CallPlan:
        """
        Produce the plan for one upcoming call.

        Parameters
        ----------
        patient_id        Namespace for history and profile lookups.
        protocol_items    Required items; defaults to STANDARD_PROTOCOL_ITEMS.
        retrieval_query   Semantic query for history. None retrieves the
                          most recent calls instead.
        n_history         How many summaries to retrieve.
        carry_over_symptoms
                          Symptom descriptions left underspecified by a
                          previous call. Each is run through the probe
                          checklist (agents/symptom_probe.py) and the
                          resulting questions are asked first.
        extra_context     Free text appended to the prompt (e.g. a discharge
                          note excerpt).

        Never raises on API failure: returns a deterministic fallback plan
        with `fallback_used=True` instead.
        """
        notes: list[str] = []
        profile = self._load_profile(patient_id, notes)
        history = self._load_history(patient_id, retrieval_query, n_history, notes)
        tone_directives = derive_tone_directives(profile)
        carry_over, probe_plans = self._carry_over_questions(carry_over_symptoms)

        personalization_used = not profile.use_default_script
        base = dict(
            patient_id=patient_id,
            tone_directives=tone_directives,
            personalization_used=personalization_used,
            profile_confidence=profile.confidence,
            profile_call_count=profile.call_count,
            history_call_ids=[str(getattr(h, "call_id", "")) for h in history],
            model=self._model,
        )

        user_prompt = _build_user_prompt(
            patient_id=patient_id,
            profile=profile,
            history=history,
            protocol=protocol_items,
            carry_over=carry_over,
            tone_directives=tone_directives,
            clinic_name=self._clinic_name,
            extra_context=extra_context,
        )

        try:
            raw, usage = self._call_llm(user_prompt)
        except Exception as exc:  # noqa: BLE001 - degradation must be total
            notes.append(f"LLM call failed ({type(exc).__name__}: {exc}); used fallback plan.")
            return self._fallback_plan(protocol_items, carry_over, notes, **base)

        self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._call_count += 1

        data = _parse_json(raw)
        if data is None:
            notes.append("Model returned unparseable JSON; used fallback plan.")
            plan = self._fallback_plan(protocol_items, carry_over, notes, **base)
            plan.token_usage = self.token_usage
            return plan

        questions = _to_planned_questions(data.get("questions"), carry_over)
        questions = _enforce_mandatory(questions, protocol_items, notes)

        continuity = [str(t) for t in (data.get("continuity_topics") or []) if str(t).strip()]
        if not personalization_used and continuity:
            # Deterministic gate: low profile confidence means no
            # conversational callbacks, whatever the model produced.
            notes.append(
                f"Dropped {len(continuity)} continuity topic(s): profile "
                f"confidence {profile.confidence:.2f} is below "
                f"{PERSONALIZATION_CONFIDENCE_THRESHOLD:.2f}."
            )
            continuity = []

        opening = str(data.get("opening") or "").strip()
        if not opening:
            opening = DEFAULT_OPENING.format(clinic=self._clinic_name)
            notes.append("Model returned no opening; used the neutral default.")
        closing = str(data.get("closing") or "").strip() or DEFAULT_CLOSING

        plan = CallPlan(
            opening=opening,
            questions=questions,
            continuity_topics=continuity,
            closing=closing,
            notes=notes,
            token_usage=self.token_usage,
            **base,
        )
        if probe_plans:
            plan.notes.append(
                f"Carried over {len(carry_over)} probe question(s) from "
                f"{len(probe_plans)} unresolved symptom(s)."
            )
        return plan

    # ------------------------------------------------------------------
    # Input loading (each degrades independently)
    # ------------------------------------------------------------------

    def _load_profile(self, patient_id: str, notes: list[str]) -> PersonalizationProfile:
        """
        Fetch the communication profile, defaulting to zero confidence.

        A missing or broken store must not upgrade the patient to
        personalised handling, so every failure path lands on
        use_default_script=True.
        """
        if self._profile_store is None:
            return _zero_profile(patient_id)
        try:
            return self._profile_store.get_profile(patient_id)
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"Profile lookup failed ({type(exc).__name__}); "
                f"using neutral default script."
            )
            return _zero_profile(patient_id)

    def _load_history(
        self,
        patient_id: str,
        retrieval_query: Optional[str],
        n_history: int,
        notes: list[str],
    ) -> list[Any]:
        """Retrieve call summaries; an empty list is a valid outcome."""
        if self._rag is None or n_history <= 0:
            return []
        try:
            if retrieval_query:
                return list(self._rag.retrieve(patient_id, retrieval_query,
                                               n_results=n_history))
            return list(self._rag.retrieve_recent(patient_id, n=n_history))
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"History retrieval failed ({type(exc).__name__}); "
                f"planning without previous calls."
            )
            return []

    @staticmethod
    def _carry_over_questions(
        carry_over_symptoms: Optional[Iterable[str]],
    ) -> tuple[list[str], list[SymptomProbePlan]]:
        """
        Turn last call's underspecified symptoms into probe questions.

        Reuses the same checklist the in-call probe uses, so a symptom that
        went unclarified is picked up with identical wording next time.
        """
        questions: list[str] = []
        plans: list[SymptomProbePlan] = []
        for text in carry_over_symptoms or []:
            probe = build_probe_plan("symptom_update", text)
            if not probe.has_questions:
                continue
            plans.append(probe)
            for q in probe.questions:
                if q not in questions:
                    questions.append(q)
        return questions, plans

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str):
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content, response.usage

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_plan(
        self,
        protocol_items: Sequence[ProtocolItem],
        carry_over: Sequence[str],
        notes: list[str],
        **base: Any,
    ) -> CallPlan:
        """
        The deterministic plan: neutral opening, template questions, no
        personalisation. Always covers every mandatory protocol item.
        """
        questions = [
            PlannedQuestion(
                field_name="followup",
                question=q,
                rationale="Carried over from an unresolved symptom in the previous call.",
                priority=1,
                from_history=True,
            )
            for q in carry_over
        ]
        questions += [
            PlannedQuestion(
                field_name=item.field_name,
                question=DEFAULT_QUESTIONS.get(
                    item.field_name,
                    f"Can you tell me a bit about {item.label}?",
                ),
                rationale="Fixed protocol template (fallback plan).",
                priority=2,
            )
            for item in protocol_items
        ]
        base = dict(base)
        # A fallback plan is never personalised, whatever the profile says.
        base["personalization_used"] = False
        base["tone_directives"] = list(NEUTRAL_TONE_DIRECTIVES)
        return CallPlan(
            opening=DEFAULT_OPENING.format(clinic=self._clinic_name),
            questions=questions,
            continuity_topics=[],
            closing=DEFAULT_CLOSING,
            fallback_used=True,
            notes=notes,
            token_usage=self.token_usage,
            **base,
        )

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    @property
    def token_usage(self) -> dict[str, Any]:
        """Cumulative usage across every plan_call() on this instance."""
        prompt_price, completion_price = PRICING_USD_PER_1M.get(self._model, (0.0, 0.0))
        cost = (
            self._prompt_tokens / 1_000_000 * prompt_price
            + self._completion_tokens / 1_000_000 * completion_price
        )
        return {
            "model": self._model,
            "calls": self._call_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "estimated_cost_usd": round(cost, 6),
        }

    def print_token_usage(self, label: str = "CallPlanner token usage") -> dict[str, Any]:
        """Print the usage table and return it, for scripts and demos."""
        usage = self.token_usage
        width = 52
        print("=" * width)
        print(label)
        print("-" * width)
        print(f"  model              : {usage['model']}")
        print(f"  llm calls          : {usage['calls']}")
        print(f"  prompt tokens      : {usage['prompt_tokens']:,}")
        print(f"  completion tokens  : {usage['completion_tokens']:,}")
        print(f"  total tokens       : {usage['total_tokens']:,}")
        print(f"  estimated cost     : ${usage['estimated_cost_usd']:.6f} USD")
        if usage["calls"]:
            print(
                f"  avg per call       : "
                f"{usage['total_tokens'] / usage['calls']:,.0f} tokens / "
                f"${usage['estimated_cost_usd'] / usage['calls']:.6f}"
            )
        print("=" * width)
        return usage


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------

def _parse_json(raw: Optional[str]) -> Optional[dict]:
    """Parse the model's JSON; None on anything unusable."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _to_planned_questions(
    raw_questions: Any,
    carry_over: Sequence[str] = (),
) -> list[PlannedQuestion]:
    """
    Convert the model's question array into PlannedQuestion objects.

    Malformed entries are skipped rather than raising - a partial plan is
    repaired by _enforce_mandatory, while an exception here would lose the
    whole plan.
    """
    if not isinstance(raw_questions, list):
        return []

    carry_over_set = set(carry_over)
    questions: list[PlannedQuestion] = []
    for entry in raw_questions:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("question") or "").strip()
        if not text:
            continue
        name = str(entry.get("field_name") or "followup").strip() or "followup"
        try:
            priority = int(entry.get("priority", 2))
        except (TypeError, ValueError):
            priority = 2
        questions.append(
            PlannedQuestion(
                field_name=name,
                question=text,
                rationale=str(entry.get("rationale") or ""),
                priority=min(3, max(1, priority)),
                from_history=text in carry_over_set or name == "followup",
            )
        )
    return questions


def _enforce_mandatory(
    questions: list[PlannedQuestion],
    protocol: Sequence[ProtocolItem],
    notes: list[str],
) -> list[PlannedQuestion]:
    """
    Guarantee every mandatory protocol item is present exactly once.

    Missing items are appended from DEFAULT_QUESTIONS and recorded in
    `notes`; duplicates of the same field are dropped, keeping the first.
    This is the reason a dropped item cannot silently cost a patient their
    medication-adherence check.
    """
    deduped: list[PlannedQuestion] = []
    seen: set[str] = set()
    for q in questions:
        if q.field_name != "followup" and q.field_name in seen:
            notes.append(f"Dropped duplicate question for '{q.field_name}'.")
            continue
        seen.add(q.field_name)
        deduped.append(q)

    for item in protocol:
        if not item.mandatory or item.field_name in seen:
            continue
        notes.append(
            f"Model omitted mandatory item '{item.field_name}'; "
            f"appended the protocol template."
        )
        deduped.append(
            PlannedQuestion(
                field_name=item.field_name,
                question=DEFAULT_QUESTIONS.get(
                    item.field_name,
                    f"Can you tell me a bit about {item.label}?",
                ),
                rationale="Restored by the planner: required by the protocol.",
                priority=2,
            )
        )
    return deduped


def _zero_profile(patient_id: str) -> PersonalizationProfile:
    """A no-history profile: zero confidence, neutral script."""
    return PersonalizationProfile(
        patient_id=patient_id,
        call_count=0,
        confidence=0.0,
        use_default_script=True,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
# The in-call helpers are re-exported here so callers have one import point
# for the whole pre-call / in-call agent layer.

__all__ = [
    "CallPlanner",
    "CallPlan",
    "PlannedQuestion",
    "ProtocolItem",
    "STANDARD_PROTOCOL_ITEMS",
    "DEFAULT_QUESTIONS",
    "MODEL",
    "derive_tone_directives",
    # Tier 1 - confirmation
    "ConfirmationTracker",
    "decide_confirmation",
    # Tier 2 - symptom probing
    "build_probe_plan",
    "classify_ambiguity",
    "plan_for_claim",
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Plan one call against the real API and print the script plus token cost.

    Run: python -m agents.planner
    """
    from memory.profile_store import ProfileStore
    from arbitration.models import CallSignals

    # Six recorded calls -> confidence 0.60 -> personalisation enabled.
    store = ProfileStore()
    for _ in range(6):
        store.update("PAT-DEMO", CallSignals(
            avg_response_latency_ms=2800.0,
            pause_frequency=3.4,
            interruption_count=0,
            call_duration_s=210.0,
            speech_rate_wpm=95.0,
        ))

    class _StubRAG:
        """Stands in for CallHistoryRAG so the demo needs no ChromaDB."""

        @classmethod
        def retrieve(cls, patient_id, query, n_results=3):
            return cls._summaries(patient_id)[:n_results]

        @classmethod
        def retrieve_recent(cls, patient_id, n=3):
            return cls._summaries(patient_id)[:n]

        @staticmethod
        def _summaries(patient_id):
            from memory.call_history_rag import CallSummaryResult
            return [CallSummaryResult(
                patient_id=patient_id,
                call_id="CALL-2026-08-20",
                summary=(
                    "Week 3 post-op follow-up. Incision healing well, no "
                    "discharge. Patient reported ongoing right knee pain "
                    "when climbing stairs, rated 4 out of 10. Taking "
                    "metformin as prescribed with no missed doses. "
                    "September review appointment booked."
                ),
                timestamp="2026-08-20T10:30:00Z",
                topics=["wound_healing", "knee_pain", "medication"],
            )]

    planner = CallPlanner(rag=_StubRAG(), profile_store=store,
                          clinic_name="Riverside Clinic")
    plan = planner.plan_call(
        "PAT-DEMO",
        retrieval_query="wound healing and knee pain",
        carry_over_symptoms=["my knee hurts"],
    )

    print(f"\nPersonalisation : {plan.personalization_used} "
          f"(confidence {plan.profile_confidence:.2f}, "
          f"{plan.profile_call_count} calls)")
    print(f"Fallback used   : {plan.fallback_used}")
    print("\nTONE DIRECTIVES")
    for d in plan.tone_directives:
        print(f"  - {d}")
    print(f"\nOPENING\n  {plan.opening}")
    print("\nQUESTIONS")
    for i, q in enumerate(plan.questions, 1):
        flag = " [history]" if q.from_history else ""
        print(f"  {i}. (P{q.priority} {q.field_name}{flag}) {q.question}")
    if plan.continuity_topics:
        print(f"\nCONTINUITY TOPICS\n  {', '.join(plan.continuity_topics)}")
    print(f"\nCLOSING\n  {plan.closing}")
    if plan.notes:
        print("\nNOTES")
        for n in plan.notes:
            print(f"  - {n}")
    print(f"\nMissing mandatory items: {plan.missing_mandatory() or 'none'}\n")

    planner.print_token_usage()


if __name__ == "__main__":
    _demo()
