"""
Tests for CallPlanner (the pre-call script).

No network: an OpenAI-shaped stub client is injected, so these tests cover
the parts that are actually ours - prompt assembly, the deterministic gates
on personalisation, protocol enforcement, and degradation.

Headline scenarios
------------------
A - Low profile confidence forces the neutral script
    A one-call profile is not evidence, so personalisation is off, the
    pacing directives are the neutral defaults, and any continuity topics
    the model produced are stripped in code rather than trusted.

B - High profile confidence allows personalisation
    Pacing directives are derived from the measured signals, and the model
    is allowed to pick up previous topics.

C - Degradation never loses the call
    An API exception or unparseable JSON still yields a usable plan
    covering every mandatory protocol item, with fallback_used=True.

D - Protocol enforcement is not left to the model
    A dropped mandatory item is restored from the template; a duplicated
    one is removed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.models import (
    STANDARD_PROTOCOL_ITEMS,
    PlannedQuestion,
    ProtocolItem,
)
from agents.planner import (
    DEFAULT_QUESTIONS,
    HIGH_INTERRUPTION_COUNT,
    NEUTRAL_TONE_DIRECTIVES,
    SLOW_RESPONSE_LATENCY_MS,
    SLOW_SPEECH_RATE_WPM,
    CallPlanner,
    derive_tone_directives,
)
from memory.profile_store import PersonalizationProfile

PATIENT = "PAT-001"

FULL_PAYLOAD = {
    "opening": "Hello Mr Smith, it's the follow-up team at the clinic.",
    "questions": [
        {"field_name": "symptom_update",
         "question": "Any new symptoms since we last spoke?",
         "rationale": "Protocol item.", "priority": 1},
        {"field_name": "medication_adherence",
         "question": "Are you taking your tablets as prescribed?",
         "rationale": "Protocol item.", "priority": 1},
        {"field_name": "pain_level",
         "question": "Where is your pain on a scale of 0 to 10?",
         "rationale": "Protocol item.", "priority": 2},
        {"field_name": "appointment_compliance",
         "question": "Did you get to your appointment last week?",
         "rationale": "Protocol item.", "priority": 2},
        {"field_name": "emergency_concerns",
         "question": "Has anything worried you enough to call someone?",
         "rationale": "Protocol item.", "priority": 1},
    ],
    "continuity_topics": ["knee pain", "son's wedding"],
    "closing": "Thanks for your time, take care.",
}


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubClient:
    """Minimal stand-in for the OpenAI client's chat.completions.create."""

    def __init__(self, payload=None, raw=None, exc=None,
                 prompt_tokens=800, completion_tokens=200):
        self._raw = raw if raw is not None else json.dumps(payload or FULL_PAYLOAD)
        self._exc = exc
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._raw))],
            usage=SimpleNamespace(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
            ),
        )

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1]["messages"][1]["content"]


class _StubProfileStore:
    def __init__(self, profile, exc=None):
        self._profile = profile
        self._exc = exc

    def get_profile(self, patient_id):
        if self._exc is not None:
            raise self._exc
        return self._profile


class _StubRAG:
    """Duck-typed CallHistoryRAG (avoids importing chromadb in unit tests)."""

    def __init__(self, summaries=None, exc=None):
        self._summaries = summaries if summaries is not None else [_summary()]
        self._exc = exc
        self.retrieve_calls: list[tuple] = []
        self.recent_calls: list[tuple] = []

    def retrieve(self, patient_id, query, n_results=3):
        if self._exc is not None:
            raise self._exc
        self.retrieve_calls.append((patient_id, query, n_results))
        return self._summaries[:n_results]

    def retrieve_recent(self, patient_id, n=5):
        if self._exc is not None:
            raise self._exc
        self.recent_calls.append((patient_id, n))
        return self._summaries[:n]


def _summary(call_id="CALL-2026-08-20", topics=("wound_healing", "knee_pain")):
    return SimpleNamespace(
        patient_id=PATIENT,
        call_id=call_id,
        summary="Week 3 post-op. Incision healing well. Right knee pain 4/10.",
        timestamp="2026-08-20T10:30:00Z",
        topics=list(topics),
        distance=0.2,
    )


def _profile(call_count=8, confidence=0.8, use_default=False, **signals):
    defaults = dict(
        avg_response_latency_ms=1400.0,
        avg_pause_frequency=1.1,
        avg_speech_rate_wpm=140.0,
        avg_interruption_count=0.0,
        avg_call_duration_s=300.0,
    )
    defaults.update(signals)
    return PersonalizationProfile(
        patient_id=PATIENT,
        call_count=call_count,
        confidence=confidence,
        use_default_script=use_default,
        **defaults,
    )


LOW_CONFIDENCE_PROFILE = _profile(call_count=1, confidence=0.1, use_default=True)
HIGH_CONFIDENCE_PROFILE = _profile()


def _planner(client=None, profile=HIGH_CONFIDENCE_PROFILE, rag=None, **kw):
    return CallPlanner(
        client=client if client is not None else _StubClient(),
        profile_store=_StubProfileStore(profile) if profile is not None else None,
        rag=rag,
        **kw,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPlanCall:

    def test_plan_has_opening_questions_and_closing(self):
        plan = _planner().plan_call(PATIENT)

        assert plan.opening.startswith("Hello Mr Smith")
        assert plan.closing == "Thanks for your time, take care."
        assert len(plan.questions) == 5
        assert all(isinstance(q, PlannedQuestion) for q in plan.questions)

    def test_every_mandatory_item_is_covered(self):
        plan = _planner().plan_call(PATIENT)
        assert plan.missing_mandatory() == []
        assert plan.covered_fields == {p.field_name for p in STANDARD_PROTOCOL_ITEMS}

    def test_provenance_is_recorded(self):
        rag = _StubRAG()
        plan = _planner(rag=rag).plan_call(PATIENT)

        assert plan.patient_id == PATIENT
        assert plan.model == "gpt-4o-mini"
        assert plan.profile_call_count == 8
        assert plan.profile_confidence == pytest.approx(0.8)
        assert plan.history_call_ids == ["CALL-2026-08-20"]
        assert not plan.fallback_used

    def test_priority_is_clamped_to_the_valid_range(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"][0]["priority"] = 99
        payload["questions"][1]["priority"] = -4
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        assert {q.priority for q in plan.questions} <= {1, 2, 3}

    def test_non_integer_priority_falls_back_to_normal(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"][0]["priority"] = "urgent"
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)
        assert plan.questions[0].priority == 2

    def test_custom_protocol_is_honoured(self):
        protocol = (
            ProtocolItem("weight_change", "weight change", "Weight since discharge."),
        )
        plan = _planner().plan_call(PATIENT, protocol_items=protocol)
        assert "weight_change" in plan.covered_fields
        assert plan.missing_mandatory(protocol) == []


# ---------------------------------------------------------------------------
# Scenario A / B - the personalisation gate
# ---------------------------------------------------------------------------

class TestPersonalizationGate:

    def test_low_confidence_profile_disables_personalization(self):
        plan = _planner(profile=LOW_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert not plan.personalization_used
        assert plan.tone_directives == list(NEUTRAL_TONE_DIRECTIVES)

    def test_low_confidence_strips_continuity_topics(self):
        # The model returned two topics; the gate is in code, not in the prompt.
        plan = _planner(profile=LOW_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert plan.continuity_topics == []
        assert any("continuity topic" in n for n in plan.notes)

    def test_high_confidence_keeps_continuity_topics(self):
        plan = _planner(profile=HIGH_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert plan.personalization_used
        assert plan.continuity_topics == ["knee pain", "son's wedding"]

    def test_prompt_tells_the_model_personalization_is_disabled(self):
        client = _StubClient()
        _planner(client=client, profile=LOW_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert "PERSONALIZATION: disabled" in client.last_user_prompt

    def test_unreliable_signal_values_are_withheld_from_the_prompt(self):
        # A model shown latency numbers will use them, so at low confidence
        # they are not in the prompt at all.
        client = _StubClient()
        _planner(client=client, profile=LOW_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert "avg_response_latency_ms" not in client.last_user_prompt

    def test_reliable_signal_values_are_included(self):
        client = _StubClient()
        _planner(client=client, profile=HIGH_CONFIDENCE_PROFILE).plan_call(PATIENT)

        assert "avg_response_latency_ms" in client.last_user_prompt
        assert "PERSONALIZATION: enabled" in client.last_user_prompt

    def test_missing_profile_store_falls_back_to_neutral(self):
        planner = CallPlanner(client=_StubClient(), profile_store=None)
        plan = planner.plan_call(PATIENT)

        assert not plan.personalization_used
        assert plan.profile_confidence == 0.0
        assert plan.profile_call_count == 0

    def test_broken_profile_store_falls_back_to_neutral(self):
        planner = CallPlanner(
            client=_StubClient(),
            profile_store=_StubProfileStore(None, exc=RuntimeError("db down")),
        )
        plan = planner.plan_call(PATIENT)

        assert not plan.personalization_used
        assert any("Profile lookup failed" in n for n in plan.notes)


class TestToneDirectives:

    def test_neutral_when_profile_is_untrustworthy(self):
        assert derive_tone_directives(LOW_CONFIDENCE_PROFILE) == list(
            NEUTRAL_TONE_DIRECTIVES
        )

    def test_slow_responder_gets_a_pause_directive(self):
        profile = _profile(avg_response_latency_ms=SLOW_RESPONSE_LATENCY_MS + 100)
        directives = derive_tone_directives(profile)
        assert any("needs time before answering" in d for d in directives)

    def test_interrupter_gets_a_brevity_directive(self):
        profile = _profile(avg_interruption_count=HIGH_INTERRUPTION_COUNT + 1)
        directives = derive_tone_directives(profile)
        assert any("speak over the agent" in d for d in directives)

    def test_slow_speaker_gets_a_pace_directive(self):
        profile = _profile(avg_speech_rate_wpm=SLOW_SPEECH_RATE_WPM - 10)
        directives = derive_tone_directives(profile)
        assert any("speaks slowly" in d for d in directives)

    def test_unremarkable_profile_still_gets_one_directive(self):
        directives = derive_tone_directives(HIGH_CONFIDENCE_PROFILE)
        assert len(directives) >= 1
        assert any("communicates comfortably" in d for d in directives)

    def test_missing_speech_rate_is_not_treated_as_slow(self):
        profile = _profile(avg_speech_rate_wpm=None)
        directives = derive_tone_directives(profile)
        assert not any("speaks slowly" in d for d in directives)

    def test_directives_reach_the_prompt(self):
        client = _StubClient()
        profile = _profile(avg_response_latency_ms=SLOW_RESPONSE_LATENCY_MS + 100)
        _planner(client=client, profile=profile).plan_call(PATIENT)

        assert "TONE DIRECTIVES" in client.last_user_prompt
        assert "needs time before answering" in client.last_user_prompt


# ---------------------------------------------------------------------------
# History retrieval
# ---------------------------------------------------------------------------

class TestHistoryRetrieval:

    def test_recency_retrieval_by_default(self):
        rag = _StubRAG()
        _planner(rag=rag).plan_call(PATIENT, n_history=2)

        assert rag.recent_calls == [(PATIENT, 2)]
        assert rag.retrieve_calls == []

    def test_semantic_retrieval_when_a_query_is_given(self):
        rag = _StubRAG()
        _planner(rag=rag).plan_call(PATIENT, retrieval_query="knee pain", n_history=2)

        assert rag.retrieve_calls == [(PATIENT, "knee pain", 2)]
        assert rag.recent_calls == []

    def test_summaries_reach_the_prompt(self):
        client = _StubClient()
        _planner(client=client, rag=_StubRAG()).plan_call(PATIENT)

        prompt = client.last_user_prompt
        assert "CALL-2026-08-20" in prompt
        assert "Incision healing well" in prompt

    def test_no_rag_means_first_contact_wording(self):
        client = _StubClient()
        _planner(client=client, rag=None).plan_call(PATIENT)

        assert "no previous calls on record" in client.last_user_prompt

    def test_retrieval_failure_degrades_to_no_history(self):
        rag = _StubRAG(exc=RuntimeError("chroma down"))
        plan = _planner(rag=rag).plan_call(PATIENT)

        assert plan.history_call_ids == []
        assert any("History retrieval failed" in n for n in plan.notes)
        assert not plan.fallback_used  # planning itself still succeeded

    def test_zero_history_count_skips_retrieval(self):
        rag = _StubRAG()
        _planner(rag=rag).plan_call(PATIENT, n_history=0)
        assert rag.recent_calls == []

    def test_history_is_marked_reference_only_at_low_confidence(self):
        client = _StubClient()
        _planner(client=client, rag=_StubRAG(),
                 profile=LOW_CONFIDENCE_PROFILE).plan_call(PATIENT)

        # Clinical facts stay usable; conversational callbacks do not.
        assert "Do not refer to them conversationally" in client.last_user_prompt


# ---------------------------------------------------------------------------
# Scenario D - protocol enforcement
# ---------------------------------------------------------------------------

class TestProtocolEnforcement:

    def test_omitted_mandatory_item_is_restored(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"] = [
            q for q in payload["questions"]
            if q["field_name"] != "medication_adherence"
        ]
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        assert plan.missing_mandatory() == []
        restored = next(q for q in plan.questions
                        if q.field_name == "medication_adherence")
        assert restored.question == DEFAULT_QUESTIONS["medication_adherence"]
        assert any("omitted mandatory item" in n for n in plan.notes)

    def test_duplicate_question_is_dropped(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"].append(dict(payload["questions"][0]))
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        names = [q.field_name for q in plan.questions]
        assert names.count("symptom_update") == 1
        assert any("duplicate question" in n for n in plan.notes)

    def test_empty_question_list_is_fully_repaired(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"] = []
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        assert plan.missing_mandatory() == []
        assert len(plan.questions) == len(STANDARD_PROTOCOL_ITEMS)

    def test_malformed_question_entries_are_skipped(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"] += ["not a dict", {"question": ""}, {}]
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        assert plan.missing_mandatory() == []
        assert all(q.question.strip() for q in plan.questions)

    def test_extra_history_question_is_kept_and_marked(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["questions"].append({
            "field_name": "followup",
            "question": "How is the knee getting on since last time?",
            "rationale": "Raised in the previous call.",
            "priority": 1,
        })
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        extra = [q for q in plan.questions if q.field_name == "followup"]
        assert len(extra) == 1
        assert extra[0].from_history

    def test_missing_opening_falls_back_to_the_default(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        payload["opening"] = "  "
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)

        assert "follow-up team" in plan.opening
        assert any("no opening" in n for n in plan.notes)

    def test_missing_closing_falls_back_to_the_default(self):
        payload = json.loads(json.dumps(FULL_PAYLOAD))
        del payload["closing"]
        plan = _planner(client=_StubClient(payload=payload)).plan_call(PATIENT)
        assert "Take care" in plan.closing


# ---------------------------------------------------------------------------
# Scenario C - degradation
# ---------------------------------------------------------------------------

class TestDegradation:

    def test_api_failure_yields_a_usable_fallback_plan(self):
        planner = _planner(client=_StubClient(exc=RuntimeError("503 upstream")))
        plan = planner.plan_call(PATIENT)

        assert plan.fallback_used
        assert plan.missing_mandatory() == []
        assert plan.opening
        assert plan.closing
        assert any("LLM call failed" in n for n in plan.notes)

    def test_unparseable_json_yields_a_fallback_plan(self):
        planner = _planner(client=_StubClient(raw="not json at all"))
        plan = planner.plan_call(PATIENT)

        assert plan.fallback_used
        assert plan.missing_mandatory() == []
        assert any("unparseable JSON" in n for n in plan.notes)

    def test_non_object_json_yields_a_fallback_plan(self):
        planner = _planner(client=_StubClient(raw='["a", "list"]'))
        plan = planner.plan_call(PATIENT)
        assert plan.fallback_used

    def test_fallback_uses_the_protocol_templates(self):
        planner = _planner(client=_StubClient(exc=RuntimeError("boom")))
        plan = planner.plan_call(PATIENT)

        for item in STANDARD_PROTOCOL_ITEMS:
            q = next(q for q in plan.questions if q.field_name == item.field_name)
            assert q.question == DEFAULT_QUESTIONS[item.field_name]

    def test_fallback_is_never_personalised(self):
        # Even with a trustworthy profile: the plan the model would have
        # personalised does not exist, so claiming personalisation would lie.
        planner = _planner(client=_StubClient(exc=RuntimeError("boom")),
                           profile=HIGH_CONFIDENCE_PROFILE)
        plan = planner.plan_call(PATIENT)

        assert not plan.personalization_used
        assert plan.tone_directives == list(NEUTRAL_TONE_DIRECTIVES)
        assert plan.continuity_topics == []

    def test_fallback_still_reports_the_profile_it_saw(self):
        planner = _planner(client=_StubClient(exc=RuntimeError("boom")))
        plan = planner.plan_call(PATIENT)
        assert plan.profile_call_count == 8

    def test_unparseable_json_still_counts_tokens(self):
        planner = _planner(client=_StubClient(raw="nope"))
        plan = planner.plan_call(PATIENT)
        assert plan.token_usage["total_tokens"] == 1000

    def test_api_failure_counts_no_tokens(self):
        planner = _planner(client=_StubClient(exc=RuntimeError("boom")))
        planner.plan_call(PATIENT)
        assert planner.token_usage["calls"] == 0


# ---------------------------------------------------------------------------
# Carry-over probe questions (wiring to agents/symptom_probe.py)
# ---------------------------------------------------------------------------

class TestCarryOverSymptoms:

    def test_unresolved_symptom_becomes_a_priority_question(self):
        client = _StubClient()
        _planner(client=client).plan_call(
            PATIENT, carry_over_symptoms=["my arm hurts"])

        prompt = client.last_user_prompt
        assert "CARRY-OVER QUESTIONS" in prompt
        assert "Whereabouts exactly" in prompt

    def test_carry_over_is_noted_on_the_plan(self):
        plan = _planner().plan_call(PATIENT, carry_over_symptoms=["my arm hurts"])
        assert any("Carried over" in n for n in plan.notes)

    def test_specific_symptom_produces_no_carry_over(self):
        client = _StubClient()
        plan = _planner(client=client).plan_call(
            PATIENT,
            carry_over_symptoms=["left inner forearm, constant dull ache, three days"],
        )
        assert "CARRY-OVER QUESTIONS" not in client.last_user_prompt
        assert not any("Carried over" in n for n in plan.notes)

    def test_carry_over_survives_the_fallback_path(self):
        planner = _planner(client=_StubClient(exc=RuntimeError("boom")))
        plan = planner.plan_call(PATIENT, carry_over_symptoms=["my arm hurts"])

        carried = [q for q in plan.questions if q.from_history]
        assert carried
        assert all(q.priority == 1 for q in carried)
        assert plan.missing_mandatory() == []

    def test_duplicate_carry_over_text_is_deduplicated(self):
        client = _StubClient()
        _planner(client=client).plan_call(
            PATIENT, carry_over_symptoms=["my arm hurts", "my arm hurts"])

        prompt = client.last_user_prompt
        assert prompt.count("Whereabouts exactly") == 1


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

class TestPromptAssembly:

    def test_protocol_items_are_listed(self):
        client = _StubClient()
        _planner(client=client).plan_call(PATIENT)

        prompt = client.last_user_prompt
        assert "MANDATORY PROTOCOL ITEMS" in prompt
        for item in STANDARD_PROTOCOL_ITEMS:
            assert item.field_name in prompt

    def test_clinic_name_is_passed_through(self):
        client = _StubClient()
        _planner(client=client, clinic_name="Riverside Clinic").plan_call(PATIENT)
        assert "Riverside Clinic" in client.last_user_prompt

    def test_extra_context_is_appended(self):
        client = _StubClient()
        _planner(client=client).plan_call(
            PATIENT, extra_context="Discharged 2026-09-01 after knee replacement.")
        assert "knee replacement" in client.last_user_prompt

    def test_json_mode_and_temperature_are_set(self):
        client = _StubClient()
        _planner(client=client).plan_call(PATIENT)

        kwargs = client.calls[-1]
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == "gpt-4o-mini"
        assert 0.0 <= kwargs["temperature"] <= 1.0

    def test_system_prompt_forbids_giving_advice(self):
        client = _StubClient()
        _planner(client=client).plan_call(PATIENT)

        system = client.calls[-1]["messages"][0]["content"]
        assert "Ask; never advise" in system
        assert "no diagnosis" in system.lower()


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

class TestTokenUsage:

    def test_usage_is_reported_on_the_plan(self):
        plan = _planner().plan_call(PATIENT)

        assert plan.token_usage["calls"] == 1
        assert plan.token_usage["prompt_tokens"] == 800
        assert plan.token_usage["completion_tokens"] == 200
        assert plan.token_usage["total_tokens"] == 1000

    def test_usage_accumulates_across_calls(self):
        planner = _planner()
        planner.plan_call(PATIENT)
        planner.plan_call(PATIENT)

        usage = planner.token_usage
        assert usage["calls"] == 2
        assert usage["total_tokens"] == 2000

    def test_cost_uses_the_model_price_list(self):
        planner = _planner()
        planner.plan_call(PATIENT)

        # 800 prompt @ $0.15/1M + 200 completion @ $0.60/1M
        expected = 800 / 1_000_000 * 0.15 + 200 / 1_000_000 * 0.60
        assert planner.token_usage["estimated_cost_usd"] == pytest.approx(
            round(expected, 6))

    def test_unknown_model_reports_zero_cost_rather_than_guessing(self):
        planner = CallPlanner(client=_StubClient(), model="some-new-model")
        planner.plan_call(PATIENT)
        assert planner.token_usage["estimated_cost_usd"] == 0.0

    def test_print_token_usage_outputs_a_table(self, capsys):
        planner = _planner()
        planner.plan_call(PATIENT)
        returned = planner.print_token_usage()

        out = capsys.readouterr().out
        assert "gpt-4o-mini" in out
        assert "prompt tokens" in out
        assert "estimated cost" in out
        assert "avg per call" in out
        assert returned["total_tokens"] == 1000

    def test_print_token_usage_before_any_call(self, capsys):
        planner = _planner()
        planner.print_token_usage()

        out = capsys.readouterr().out
        assert "llm calls          : 0" in out
        # No division by zero for the average line.
        assert "avg per call" not in out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_injected_client_needs_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        CallPlanner(client=_StubClient())  # must not raise

    def test_missing_api_key_is_reported_clearly(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            CallPlanner()
