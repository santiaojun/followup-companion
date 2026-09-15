"""
Tests for RealExtractor's output normalisation and clarification judgement.

No network: an OpenAI-shaped stub client is injected. The stub dispatches on
the system prompt, so one client serves both the Stage 1 extraction call and
the Stage 4 clarification call.

The architectural property under test
-------------------------------------
needs_clarification is decided in a SEPARATE call from extraction, because
asking for it inline degraded the safety-critical emergency_concerns field
(see the Stage 4 note in real_extractor.py). Three tests pin that:

  * the extraction prompt never mentions the flag
  * Stage 4 is skipped entirely when nothing could need clarifying
  * a Stage 4 failure is swallowed and leaves extraction fully intact

Plus the usual: the flag survives the round trip, "insufficient" can never
be clarifiable, and the arbitration view is untouched.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.models import AmbiguityKind
from agents.symptom_probe import classify_ambiguity
from arbitration.models import LLMEvidence
from eval.real_extractor import (
    CLARIFICATION_MODEL,
    MODEL,
    SYSTEM_PROMPT,
    RealExtractor,
)
from triage.models import EVIDENCE_SOURCES, ExtractedClaim

VAGUE = "my arm hurts"
SPECIFIC = "constant dull ache in my left inner forearm for three days"


def _field(
    field_name="symptom_update",
    value=VAGUE,
    confidence=0.95,
    evidence_source="direct_quote",
    candidates=None,
    citations=(VAGUE,),
    reasoning="Patient stated it directly.",
):
    return {
        "field_name": field_name,
        "evidence_source": evidence_source,
        "extracted_value": value,
        "candidates": candidates,
        "citations": list(citations),
        "confidence": confidence,
        "reasoning": reasoning,
    }


class _StubClient:
    """
    Stand-in for client.chat.completions.create, for both stages.

    Dispatches on the system prompt: the Stage 4 prompt is the one that asks
    for "judgements".
    """

    def __init__(
        self,
        fields=None,
        raw=None,
        judgements=None,
        clarify_raw=None,
        clarify_exc=None,
        prompt_tokens=500,
        completion_tokens=300,
        clarify_prompt_tokens=90,
        clarify_completion_tokens=20,
    ):
        self._extract_raw = raw if raw is not None else json.dumps(
            {"fields": fields if fields is not None else [_field()]})
        self._judgements = judgements
        self._clarify_raw = clarify_raw
        self._clarify_exc = clarify_exc
        self._pt, self._ct = prompt_tokens, completion_tokens
        self._cpt, self._cct = clarify_prompt_tokens, clarify_completion_tokens

        self.extract_calls: list[dict] = []
        self.clarify_calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "judgements" in system:
            self.clarify_calls.append(kwargs)
            if self._clarify_exc is not None:
                raise self._clarify_exc
            if self._clarify_raw is not None:
                body = self._clarify_raw
            else:
                judgements = self._judgements
                if judgements is None:
                    judgements = [{"field_name": "symptom_update",
                                   "needs_clarification": True}]
                body = json.dumps({"judgements": judgements})
            return self._response(body, self._cpt, self._cct)

        self.extract_calls.append(kwargs)
        return self._response(self._extract_raw, self._pt, self._ct)

    @staticmethod
    def _response(content, pt, ct):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
        )

    @property
    def clarify_payload(self) -> dict:
        return json.loads(self.clarify_calls[-1]["messages"][1]["content"])


def _extractor(**kw):
    clarify = kw.pop("clarify", True)
    return RealExtractor(client=_StubClient(**kw), clarify=clarify)


# ---------------------------------------------------------------------------
# The safety property: extraction is not perturbed
# ---------------------------------------------------------------------------

class TestExtractionIsolation:

    def test_extraction_prompt_never_mentions_the_flag(self):
        """
        The Stage 1 prompt must stay byte-identical to the version the risk
        baseline was measured on. Asking for needs_clarification inline made
        the model return "None" for emergency_concerns on veiled suicidal
        ideation; keeping it out is the fix.
        """
        assert "needs_clarification" not in SYSTEM_PROMPT
        assert "CLARIFICATION" not in SYSTEM_PROMPT

    def test_clarification_uses_a_second_call(self):
        client = _StubClient(fields=[_field()])
        RealExtractor(client=client).extract("transcript", {})

        assert len(client.extract_calls) == 1
        assert len(client.clarify_calls) == 1
        assert client.clarify_calls[0]["model"] == CLARIFICATION_MODEL

    def test_clarification_failure_does_not_break_extraction(self):
        client = _StubClient(fields=[_field()],
                             clarify_exc=RuntimeError("503 upstream"))
        ex = RealExtractor(client=client)
        evidence, raw = ex.extract("transcript", {})

        # Extraction is complete and usable...
        assert len(evidence) == 1
        assert evidence[0].extracted_value == VAGUE
        assert raw["_model"] == MODEL
        # ...and the claim simply falls back to the keyword path.
        assert ex.last_claims[0].needs_clarification is False

    def test_clarify_can_be_disabled(self):
        client = _StubClient(fields=[_field()])
        ex = RealExtractor(client=client, clarify=False)
        ex.extract("transcript", {})

        assert client.clarify_calls == []
        assert ex.last_claims[0].needs_clarification is False


# ---------------------------------------------------------------------------
# Skipping Stage 4 when there is nothing to judge
# ---------------------------------------------------------------------------

class TestStage4Skipping:

    @pytest.mark.parametrize("value", [
        "no new symptoms", "nothing", "none", "I'm fine", "6", "7/10",
        "", None,
    ])
    def test_denials_and_scores_skip_the_call(self, value):
        client = _StubClient(fields=[_field(value=value)])
        RealExtractor(client=client).extract("transcript", {})
        assert client.clarify_calls == [], f"{value!r} should not be judged"

    def test_non_symptom_fields_skip_the_call(self):
        # emergency_concerns is deliberately excluded: an urgent concern is
        # escalated to a human, not probed by the agent.
        fields = [_field(field_name="emergency_concerns", value="chest pain"),
                  _field(field_name="medication_adherence", value="yes"),
                  _field(field_name="appointment_compliance", value="went")]
        client = _StubClient(fields=fields)
        RealExtractor(client=client).extract("transcript", {})
        assert client.clarify_calls == []

    def test_insufficient_evidence_skips_the_call(self):
        client = _StubClient(fields=[_field(evidence_source="insufficient",
                                            value=None, citations=())])
        RealExtractor(client=client).extract("transcript", {})
        assert client.clarify_calls == []

    def test_vague_symptom_does_trigger_the_call(self):
        client = _StubClient(fields=[_field(value=VAGUE)])
        RealExtractor(client=client).extract("transcript", {})
        assert len(client.clarify_calls) == 1

    def test_pain_level_is_in_scope(self):
        client = _StubClient(
            fields=[_field(field_name="pain_level", value="it's quite sore")],
            judgements=[{"field_name": "pain_level",
                         "needs_clarification": True}])
        ex = RealExtractor(client=client)
        ex.extract("transcript", {})

        assert len(client.clarify_calls) == 1
        assert ex.last_claims[0].needs_clarification is True

    def test_payload_carries_value_and_quotes(self):
        client = _StubClient(fields=[_field(
            value=VAGUE, citations=("my arm hurts", "just sore", "x", "y"))])
        RealExtractor(client=client).extract("transcript", {})

        item = client.clarify_payload["fields"][0]
        assert item["field_name"] == "symptom_update"
        assert item["extracted_value"] == VAGUE
        # Quotes let the judge see detail the summary dropped; capped at 3.
        assert item["patient_quotes"] == ["my arm hurts", "just sore", "x"]


# ---------------------------------------------------------------------------
# The flag, end to end
# ---------------------------------------------------------------------------

class TestNeedsClarificationRoundTrip:

    def test_true_judgement_reaches_the_claim(self):
        ex = _extractor(fields=[_field(value=VAGUE)],
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": True}])
        claims, _ = ex.extract_claims("transcript")

        assert len(claims) == 1
        assert isinstance(claims[0], ExtractedClaim)
        assert claims[0].needs_clarification is True

    def test_false_judgement_reaches_the_claim(self):
        ex = _extractor(fields=[_field(value=SPECIFIC)],
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": False}])
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is False

    def test_flag_drives_the_probe_decision(self):
        ex = _extractor(fields=[_field(value=VAGUE)],
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": True}])
        claims, _ = ex.extract_claims("transcript")
        assert classify_ambiguity(claims[0]) is AmbiguityKind.UNDERSPECIFIED

    def test_unflagged_specific_description_stays_quiet(self):
        ex = _extractor(fields=[_field(value=SPECIFIC)],
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": False}])
        claims, _ = ex.extract_claims("transcript")
        assert classify_ambiguity(claims[0]) is AmbiguityKind.NONE

    def test_judgement_only_applies_to_the_named_field(self):
        fields = [_field(field_name="symptom_update", value=VAGUE),
                  _field(field_name="pain_level", value="quite sore")]
        ex = _extractor(fields=fields,
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": True}])
        claims, _ = ex.extract_claims("transcript")
        by_name = {c.field_name: c for c in claims}

        assert by_name["symptom_update"].needs_clarification is True
        assert by_name["pain_level"].needs_clarification is False

    def test_last_claims_is_populated_by_plain_extract(self):
        ex = _extractor(fields=[_field(value=VAGUE)])
        ex.extract("transcript", {})
        assert ex.last_claims[0].needs_clarification is True

    def test_last_claims_is_empty_before_extraction(self):
        assert _extractor().last_claims == []

    def test_last_claims_returns_a_copy(self):
        ex = _extractor()
        ex.extract("transcript", {})
        ex.last_claims.clear()
        assert ex.last_claims, "internal state must not be mutable from outside"


# ---------------------------------------------------------------------------
# Rules enforced in code, not left to a model
# ---------------------------------------------------------------------------

class TestInsufficientEvidenceOverride:

    def test_insufficient_can_never_be_clarifiable(self):
        # Even if a judgement arrives for it: nothing was captured, so there
        # is no vague description to refine.
        ex = _extractor(
            fields=[_field(evidence_source="insufficient", value=None,
                           citations=())],
            judgements=[{"field_name": "symptom_update",
                         "needs_clarification": True}])
        claims, _ = ex.extract_claims("transcript")

        assert claims[0].evidence_source == "insufficient"
        assert claims[0].needs_clarification is False

    def test_insufficient_claim_routes_to_the_confirmation_path(self):
        ex = _extractor(fields=[_field(evidence_source="insufficient",
                                       value=None, confidence=0.0,
                                       citations=())])
        claims, _ = ex.extract_claims("transcript")
        assert classify_ambiguity(claims[0]) is AmbiguityKind.TRANSCRIPTION_UNCLEAR


class TestMalformedJudgements:

    def test_unparseable_response_degrades_to_false(self):
        ex = _extractor(fields=[_field(value=VAGUE)], clarify_raw="not json")
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is False

    def test_non_object_response_degrades_to_false(self):
        ex = _extractor(fields=[_field(value=VAGUE)], clarify_raw='["a"]')
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is False

    def test_missing_judgements_key_degrades_to_false(self):
        ex = _extractor(fields=[_field(value=VAGUE)], clarify_raw='{"other": 1}')
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is False

    @pytest.mark.parametrize("raw,expected", [
        (True, True), ("true", True), ("True", True), ("TRUE", True),
        (False, False), ("false", False), ("no", False),
        (1, False), (0, False), (None, False), ([], False),
    ])
    def test_flag_types_are_coerced(self, raw, expected):
        ex = _extractor(fields=[_field(value=VAGUE)],
                        judgements=[{"field_name": "symptom_update",
                                     "needs_clarification": raw}])
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is expected

    def test_malformed_entries_are_skipped(self):
        ex = _extractor(
            fields=[_field(value=VAGUE)],
            clarify_raw=json.dumps({"judgements": [
                "not a dict", {}, {"field_name": ""},
                {"field_name": "symptom_update", "needs_clarification": True},
            ]}))
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].needs_clarification is True

    def test_unknown_evidence_source_becomes_insufficient(self):
        ex = _extractor(fields=[_field(evidence_source="guessed")])
        claims, _ = ex.extract_claims("transcript")

        assert claims[0].evidence_source in EVIDENCE_SOURCES
        assert claims[0].evidence_source == "insufficient"


# ---------------------------------------------------------------------------
# The arbitration view must be unaffected
# ---------------------------------------------------------------------------

class TestArbitrationViewUnchanged:

    def test_extract_still_returns_llm_evidence(self):
        ex = _extractor(fields=[_field()])
        evidence, _ = ex.extract("transcript", {})

        assert all(isinstance(e, LLMEvidence) for e in evidence)
        assert not hasattr(evidence[0], "needs_clarification")

    def test_evidence_fields_are_untouched(self):
        ex = _extractor(fields=[_field(value="chest pain", confidence=0.91)])
        evidence, _ = ex.extract("transcript", {})

        assert evidence[0].field_name == "symptom_update"
        assert evidence[0].extracted_value == "chest pain"
        assert evidence[0].confidence == pytest.approx(0.91)

    def test_raw_output_metadata_reports_the_extraction_call(self):
        ex = _extractor()
        _, raw = ex.extract("transcript",
                            {"mock_raw_llm_output": {"asr_confidence": 0.62}})

        assert raw["asr_confidence"] == 0.62
        assert raw["_model"] == MODEL
        # Stage 1 tokens only - Stage 4 is not part of the arbitration input.
        assert raw["_prompt_tokens"] == 500

    def test_both_views_describe_the_same_fields(self):
        fields = [_field(field_name=n, value="no change") for n in
                  ("symptom_update", "medication_adherence", "pain_level")]
        ex = _extractor(fields=fields)
        evidence, _ = ex.extract("transcript", {})

        assert [e.field_name for e in evidence] == \
               [c.field_name for c in ex.last_claims]

    def test_duplicate_fields_are_deduplicated_in_both_views(self):
        fields = [_field(field_name="symptom_update", value="first ache"),
                  _field(field_name="symptom_update", value="second ache")]
        ex = _extractor(fields=fields)
        evidence, _ = ex.extract("transcript", {})

        assert len(evidence) == 1
        assert len(ex.last_claims) == 1
        assert evidence[0].extracted_value == "first ache"
        assert ex.last_claims[0].extracted_value == "first ache"

    def test_unnamed_fields_are_skipped_in_both_views(self):
        fields = [_field(field_name=""), _field(field_name="pain_level",
                                                value="no pain")]
        ex = _extractor(fields=fields)
        evidence, _ = ex.extract("transcript", {})

        assert [e.field_name for e in evidence] == ["pain_level"]
        assert [c.field_name for c in ex.last_claims] == ["pain_level"]


# ---------------------------------------------------------------------------
# Candidates and HITL interaction
# ---------------------------------------------------------------------------

class TestCandidates:

    def test_candidates_reach_the_claim(self):
        ex = _extractor(fields=[_field(candidates=[
            {"text": "stomach pain", "confidence": 0.52},
            {"text": "chest pain", "confidence": 0.48},
        ])])
        claims, _ = ex.extract_claims("transcript")

        assert claims[0].candidates is not None
        assert len(claims[0].candidates) == 2
        assert claims[0].needs_hitl  # gap 0.04 < CANDIDATE_CLOSENESS_THRESHOLD

    def test_empty_candidate_list_becomes_none(self):
        # ExtractedClaim treats None as "unambiguous"; an empty list would
        # read as "candidates exist but there are zero of them".
        ex = _extractor(fields=[_field(candidates=[])])
        claims, _ = ex.extract_claims("transcript")
        assert claims[0].candidates is None

    def test_tied_candidates_outrank_the_clarification_flag(self):
        # Both problems at once: settle the wording before asking for detail.
        ex = _extractor(fields=[_field(
            confidence=0.95,
            candidates=[{"text": "arm pain", "confidence": 0.95},
                        {"text": "palm pain", "confidence": 0.93}])])
        claims, _ = ex.extract_claims("transcript")

        assert claims[0].needs_clarification is True
        assert classify_ambiguity(claims[0]) is AmbiguityKind.TRANSCRIPTION_UNCLEAR


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

class TestPlumbing:

    def test_deterministic_settings_on_both_calls(self):
        client = _StubClient(fields=[_field(value=VAGUE)])
        RealExtractor(client=client).extract("transcript", {})

        for kwargs in client.extract_calls + client.clarify_calls:
            assert kwargs["temperature"] == 0.0
            assert kwargs["response_format"] == {"type": "json_object"}

    def test_injected_client_needs_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        RealExtractor(client=_StubClient())  # must not raise

    def test_missing_api_key_still_reported(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            RealExtractor()

    def test_unparseable_extraction_yields_no_fields(self):
        ex = RealExtractor(client=_StubClient(raw="not json"))
        evidence, _ = ex.extract("transcript", {})

        assert evidence == []
        assert ex.last_claims == []

    def test_token_usage_separates_the_two_stages(self):
        ex = _extractor(fields=[_field(value=VAGUE)])
        ex.extract("transcript", {})

        usage = ex.token_usage
        assert usage["extraction_calls"] == 1
        assert usage["clarification_calls"] == 1
        assert usage["calls"] == 2
        assert usage["prompt_tokens"] == 500 + 90
        assert usage["completion_tokens"] == 300 + 20
        assert usage["total_tokens"] == 910

    def test_skipped_stage4_costs_nothing(self):
        ex = _extractor(fields=[_field(value="no new symptoms")])
        ex.extract("transcript", {})

        usage = ex.token_usage
        assert usage["clarification_calls"] == 0
        assert usage["calls"] == 1
        assert usage["total_tokens"] == 800

    def test_failed_stage4_is_not_counted_as_a_call(self):
        ex = _extractor(fields=[_field(value=VAGUE)],
                        clarify_exc=RuntimeError("boom"))
        ex.extract("transcript", {})
        assert ex.token_usage["clarification_calls"] == 0
