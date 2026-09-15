"""
Test suite – Ambiguous / needs-follow-up scenarios.

These calls don't contain risk signals but fail quality gates:
  - Incomplete extraction (missing required fields)
  - Low LLM confidence (hedged / unclear patient responses)
  - Untraceable evidence (LLM hallucinated citations)
  - OOD / wrong-number call → REJECT

Expected outcomes: TRIAGE_AMBIGUOUS or REJECT (never ACCEPT or TRIAGE_RISK).
"""
import copy

import pytest

from arbitration import Action, ArbitrationGate
from arbitration.models import ArbitrationInput, DimensionName, LLMEvidence
from arbitration.tests.conftest import (
    NORMAL_SIGNALS,
    NORMAL_EVIDENCE,
    NORMAL_TRANSCRIPT,
    build_normal_input,
)


@pytest.fixture
def gate() -> ArbitrationGate:
    return ArbitrationGate()


# ---------------------------------------------------------------------------
# 1. Incomplete extraction – only 2 of 5 required fields covered
# ---------------------------------------------------------------------------

class TestIncompleteExtraction:
    def test_action_is_triage_ambiguous(self, gate):
        inp = build_normal_input("CALL-AMB-001")
        # Keep only 2 of the 5 required-field evidence items
        inp.llm_evidence = [
            ev for ev in NORMAL_EVIDENCE
            if ev.field_name in ("symptom_update", "pain_level")
        ]
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_passed_is_false(self, gate):
        inp = build_normal_input("CALL-AMB-001")
        inp.llm_evidence = [ev for ev in NORMAL_EVIDENCE if ev.field_name == "pain_level"]
        result = gate.evaluate(inp)
        assert result.passed is False

    def test_completeness_fails(self, gate):
        inp = build_normal_input("CALL-AMB-001")
        inp.llm_evidence = [NORMAL_EVIDENCE[0]]  # Only symptom_update
        result = gate.evaluate(inp)
        comp = result.dimensions[DimensionName.COMPLETENESS.value]
        assert not comp.passed

    def test_missing_field_flags_populated(self, gate):
        inp = build_normal_input("CALL-AMB-001")
        inp.llm_evidence = [NORMAL_EVIDENCE[0]]  # Only symptom_update
        result = gate.evaluate(inp)
        comp = result.dimensions[DimensionName.COMPLETENESS.value]
        missing = [f for f in comp.flags if f.startswith("missing_field:")]
        assert len(missing) >= 3  # At least 3 of 4 remaining fields are missing

    def test_no_risk_flags(self, gate):
        inp = build_normal_input("CALL-AMB-001")
        inp.llm_evidence = [NORMAL_EVIDENCE[0]]
        result = gate.evaluate(inp)
        assert result.risk_flags == []


# ---------------------------------------------------------------------------
# 2. Low confidence – patient gave vague / evasive answers
# ---------------------------------------------------------------------------

VAGUE_TRANSCRIPT = """
Agent: How have you been feeling since the procedure?
Patient: I don't know, kind of okay I think.
Agent: Are you taking your medications as prescribed?
Patient: Mostly, I guess, sometimes I forget.
Agent: On a scale of 1 to 10, what's your pain level?
Patient: It varies, I'm not really sure, sometimes more sometimes less.
Agent: Did you attend your follow-up appointment?
Patient: I think so, or maybe it was rescheduled.
Agent: Any emergency concerns?
Patient: I don't think so, but I'm not sure.
""".strip()

LOW_CONFIDENCE_EVIDENCE = [
    LLMEvidence(
        field_name="symptom_update",
        extracted_value="uncertain – patient unsure",
        citations=["I don't know, kind of okay I think."],
        confidence=0.35,
        reasoning="Patient was evasive; could not determine clearly.",
    ),
    LLMEvidence(
        field_name="medication_adherence",
        extracted_value="partial – sometimes forgets",
        citations=["Mostly, I guess, sometimes I forget."],
        confidence=0.40,
        reasoning="Patient acknowledged occasional non-adherence.",
    ),
    LLMEvidence(
        field_name="pain_level",
        extracted_value="variable – unclear",
        citations=["It varies, I'm not really sure, sometimes more sometimes less."],
        confidence=0.30,
        reasoning="No numeric rating given; patient could not quantify.",
    ),
    LLMEvidence(
        field_name="appointment_compliance",
        extracted_value="uncertain",
        citations=["I think so, or maybe it was rescheduled."],
        confidence=0.38,
        reasoning="Patient unsure whether appointment took place.",
    ),
    LLMEvidence(
        field_name="emergency_concerns",
        extracted_value="probably none",
        citations=["I don't think so, but I'm not sure."],
        confidence=0.45,
        reasoning="Weak denial; cannot rule out concerns.",
    ),
]


class TestLowConfidenceCall:
    def _make_inp(self, call_id: str = "CALL-AMB-CONF") -> ArbitrationInput:
        return ArbitrationInput(
            call_id=call_id,
            patient_id="PAT-VAGUE",
            transcript=VAGUE_TRANSCRIPT,
            llm_evidence=LOW_CONFIDENCE_EVIDENCE,
            call_signals=NORMAL_SIGNALS,
        )

    def test_action_is_triage_ambiguous(self, gate):
        result = gate.evaluate(self._make_inp())
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_confidence_dimension_fails(self, gate):
        result = gate.evaluate(self._make_inp())
        conf = result.dimensions[DimensionName.CONFIDENCE.value]
        assert not conf.passed

    def test_confidence_score_below_threshold(self, gate):
        result = gate.evaluate(self._make_inp())
        conf = result.dimensions[DimensionName.CONFIDENCE.value]
        assert conf.score < 0.70

    def test_no_risk_flags(self, gate):
        result = gate.evaluate(self._make_inp())
        assert result.risk_flags == []


# ---------------------------------------------------------------------------
# 3. Untraceable evidence – LLM cited text not found in transcript
# ---------------------------------------------------------------------------

class TestCitationPunctuationTolerance:
    """
    A quote terminated with punctuation the transcript does not have is
    still a real quote.

    When a model lifts a clause out of the middle of a sentence it tends to
    close the quote with a full stop:

        citation   "I have absolutely no thoughts of ending my life."
        transcript "...no thoughts of ending my life and I genuinely meant it."

    An exact substring test rejected that as untraceable, which failed 4/10
    runs on SC-025. Trailing punctuation is now stripped before matching;
    hallucinated citations still fail, because the rest must appear verbatim.
    """

    TRANSCRIPT = (
        "Agent: Any emergency concerns today?\n"
        "Patient: Nothing at all. I reassured her that I have absolutely no "
        "thoughts of ending my life and I genuinely meant it."
    )

    def _input(self, citation: str) -> ArbitrationInput:
        inp = build_normal_input("CALL-CITE-PUNCT")
        inp.transcript = self.TRANSCRIPT
        inp.llm_evidence = [
            LLMEvidence(
                field_name="emergency_concerns",
                extracted_value="no thoughts of ending my life",
                citations=[citation],
                confidence=0.95,
                reasoning="Patient denied ideation.",
            )
        ]
        return inp

    def test_appended_full_stop_is_tolerated(self, gate):
        inp = self._input("I have absolutely no thoughts of ending my life.")
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert trace.passed, trace.reason

    @pytest.mark.parametrize("suffix", ['.', ',', ';', '!', '?', '"', "'", ' ', '. '])
    def test_any_trailing_punctuation_is_tolerated(self, gate, suffix):
        inp = self._input("I have absolutely no thoughts of ending my life" + suffix)
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert trace.passed

    def test_exact_citation_still_passes(self, gate):
        inp = self._input("I have absolutely no thoughts of ending my life and I")
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert trace.passed

    def test_hallucinated_citation_still_fails(self, gate):
        # The point of the dimension: forgiving punctuation must not start
        # forgiving invention.
        inp = self._input("The patient said they felt completely hopeless.")
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert not trace.passed
        assert "untraced_field:emergency_concerns" in trace.flags

    def test_paraphrase_still_fails(self, gate):
        # Same meaning, different words - not a verbatim quote.
        inp = self._input("I definitely do not have any thoughts of ending my life.")
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert not trace.passed

    def test_punctuation_only_citation_fails_the_length_check(self, gate):
        inp = self._input("...........")
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert not trace.passed

    def test_short_quote_ending_in_a_full_stop_still_counts(self, gate):
        """
        Regression: "Yes I did." is exactly MIN_CITATION_LENGTH raw but one
        short once the period is stripped. Measuring the stripped form
        rejected a valid citation and broke SC-026 5/5, so the length gate
        stays on the citation as given.
        """
        inp = build_normal_input("CALL-CITE-SHORT")
        inp.transcript = (
            "Agent: Did you attend your follow-up appointment?\n"
            "Patient: Yes I did. The clinic was busy."
        )
        inp.llm_evidence = [
            LLMEvidence(
                field_name="appointment_compliance",
                extracted_value="attended",
                citations=["Yes I did."],
                confidence=0.95,
                reasoning="Attendance confirmed.",
            )
        ]
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert trace.passed, trace.reason

    def test_quoted_citation_is_tolerated(self, gate):
        inp = self._input('"I have absolutely no thoughts of ending my life."')
        trace = gate.evaluate(inp).dimensions[DimensionName.TRACEABILITY.value]
        assert trace.passed


class TestUntraceableEvidence:
    def test_action_is_triage_ambiguous(self, gate):
        inp = build_normal_input("CALL-AMB-TRACE")
        # Replace every citation with text that doesn't appear in transcript
        inp.llm_evidence = [
            LLMEvidence(
                field_name=ev.field_name,
                extracted_value=ev.extracted_value,
                citations=["This quote does not exist anywhere in the call recording."],
                confidence=0.85,
                reasoning=ev.reasoning,
            )
            for ev in NORMAL_EVIDENCE
        ]
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_traceability_dimension_fails(self, gate):
        inp = build_normal_input("CALL-AMB-TRACE")
        inp.llm_evidence = [
            LLMEvidence(
                field_name=ev.field_name,
                extracted_value=ev.extracted_value,
                citations=["Fabricated citation not in transcript."],
                confidence=0.85,
            )
            for ev in NORMAL_EVIDENCE
        ]
        result = gate.evaluate(inp)
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        assert not trace.passed

    def test_untraced_flags_present(self, gate):
        inp = build_normal_input("CALL-AMB-TRACE")
        inp.llm_evidence = [
            LLMEvidence(
                field_name=ev.field_name,
                extracted_value=ev.extracted_value,
                citations=["Fabricated citation not in transcript."],
                confidence=0.85,
            )
            for ev in NORMAL_EVIDENCE
        ]
        result = gate.evaluate(inp)
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        untraced = [f for f in trace.flags if f.startswith("untraced_field:")]
        assert len(untraced) == len(NORMAL_EVIDENCE)


# ---------------------------------------------------------------------------
# 4. Wrong-number / OOD call → REJECT
# ---------------------------------------------------------------------------

WRONG_NUMBER_TRANSCRIPT = """
Agent: Hello, this is a medical follow-up call for the patient.
Caller: You have the wrong number. I don't know who you are looking for.
Agent: I apologize for the inconvenience. Thank you for letting us know.
Caller: Please remove my number from your list.
""".strip()


class TestWrongNumberCall:
    def test_action_is_reject(self, gate):
        inp = ArbitrationInput(
            call_id="CALL-OOD-WN",
            patient_id="PAT-UNKNOWN",
            transcript=WRONG_NUMBER_TRANSCRIPT,
            llm_evidence=[],   # No useful extraction from a wrong-number call
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        assert result.action == Action.REJECT

    def test_ood_dimension_fails(self, gate):
        inp = ArbitrationInput(
            call_id="CALL-OOD-WN",
            patient_id="PAT-UNKNOWN",
            transcript=WRONG_NUMBER_TRANSCRIPT,
            llm_evidence=[],
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        ood = result.dimensions[DimensionName.OOD.value]
        assert not ood.passed

    def test_ood_wrong_number_flag(self, gate):
        inp = ArbitrationInput(
            call_id="CALL-OOD-WN",
            patient_id="PAT-UNKNOWN",
            transcript=WRONG_NUMBER_TRANSCRIPT,
            llm_evidence=[],
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        ood = result.dimensions[DimensionName.OOD.value]
        wn_flags = [f for f in ood.flags if "wrong_number" in f]
        assert len(wn_flags) >= 1


# ---------------------------------------------------------------------------
# 5. PII leaked into extracted fields → hard REJECT
# ---------------------------------------------------------------------------

class TestPIILeakage:
    def test_action_is_reject_on_ssn_in_extracted_value(self, gate):
        inp = build_normal_input("CALL-PII-001")
        # Simulate LLM accidentally including patient SSN in extracted value
        inp.llm_evidence = list(NORMAL_EVIDENCE) + [
            LLMEvidence(
                field_name="patient_identity",
                extracted_value="Patient SSN: 123-45-6789",   # PII violation
                citations=["The patient confirmed their identity."],
                confidence=0.99,
            )
        ]
        result = gate.evaluate(inp)
        assert result.action == Action.REJECT

    def test_pii_scrubbed_flag_set(self, gate):
        inp = build_normal_input("CALL-PII-002")
        inp.llm_evidence = list(NORMAL_EVIDENCE) + [
            LLMEvidence(
                field_name="contact",
                extracted_value="Call back at 555-867-5309",  # Phone number
                citations=["Patient said to call them back."],
                confidence=0.88,
            )
        ]
        result = gate.evaluate(inp)
        assert result.pii_scrubbed is True

    def test_pii_dimension_fails(self, gate):
        inp = build_normal_input("CALL-PII-003")
        inp.llm_evidence = list(NORMAL_EVIDENCE) + [
            LLMEvidence(
                field_name="notes",
                extracted_value="Patient email: jane.doe@example.com",
                citations=["Patient provided email."],
                confidence=0.90,
            )
        ]
        result = gate.evaluate(inp)
        pii = result.dimensions[DimensionName.PII_BOUNDARY.value]
        assert not pii.passed
        assert any("email" in f for f in pii.flags)


# ---------------------------------------------------------------------------
# 6. Traceability – extracted_value=None fields exempt from citation requirement
#    (SC-021 fix)
# ---------------------------------------------------------------------------

# Transcript for a normal post-op call that produces no emergency concerns.
NULL_CONCERNS_TRANSCRIPT = """
Agent: How have you been feeling since the surgery?
Patient: Much better, thank you. Pain is about 3 out of 10.
  I've been taking my medication every morning without missing any doses.
Agent: Great. Have you had any emergency concerns since we last spoke?
Patient: No, nothing at all. Everything feels fine.
Agent: And have you been to your follow-up appointment?
Patient: Yes, I went last Thursday. The doctor seemed happy with my progress.
""".strip()


class TestTraceabilityNullExemption:
    """
    A field with extracted_value=None makes no positive claim, so it must
    not be counted in the traceability denominator.

    Without the fix: emergency_concerns=None with no citations counted as
    untraced → 80% ratio → TRIAGE_AMBIGUOUS (SC-021 symptom).
    With the fix: exempt from denominator → 100% → ACCEPT.
    """

    def _make_null_concerns_input(self, call_id: str = "CALL-NULL-TRACE") -> ArbitrationInput:
        """Full evidence set: four fields with citations, emergency_concerns=None."""
        evidence = [
            LLMEvidence(
                field_name="symptom_update",
                extracted_value="improved – pain 3/10",
                citations=["Pain is about 3 out of 10"],
                confidence=0.92,
                reasoning="Patient reported improvement.",
            ),
            LLMEvidence(
                field_name="medication_adherence",
                extracted_value="full adherence",
                citations=["taking my medication every morning without missing any doses"],
                confidence=0.95,
                reasoning="Patient confirmed daily adherence.",
            ),
            LLMEvidence(
                field_name="appointment_compliance",
                extracted_value="attended",
                citations=["I went last Thursday"],
                confidence=0.93,
                reasoning="Patient confirmed attendance.",
            ),
            LLMEvidence(
                field_name="pain_level",
                extracted_value="3/10",
                citations=["Pain is about 3 out of 10"],
                confidence=0.94,
                reasoning="Numeric pain level stated by patient.",
            ),
            LLMEvidence(
                field_name="emergency_concerns",
                extracted_value=None,          # LLM correctly returned None — no concerns.
                citations=[],                  # No citations because nothing was reported.
                confidence=0.00,
                reasoning="Patient denied any emergency concerns.",
            ),
        ]
        return ArbitrationInput(
            call_id=call_id,
            patient_id="PAT-NULL-TRACE",
            transcript=NULL_CONCERNS_TRANSCRIPT,
            llm_evidence=evidence,
            call_signals=NORMAL_SIGNALS,
        )

    def test_null_extracted_value_field_does_not_count_against_traceability(self, gate):
        """emergency_concerns=None must not be in the traceability denominator."""
        result = gate.evaluate(self._make_null_concerns_input())
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        # 4 claimable fields, all with valid citations → 100% traceable.
        assert trace.passed, (
            f"Traceability must pass when only extracted_value=None is uncited; "
            f"score={trace.score:.2f}, reason={trace.reason!r}"
        )

    def test_action_is_accept_when_only_null_field_has_no_citation(self, gate):
        """The gate must return ACCEPT, not TRIAGE_AMBIGUOUS."""
        result = gate.evaluate(self._make_null_concerns_input())
        assert result.action == Action.ACCEPT, (
            f"Expected ACCEPT but got {result.action}. "
            f"Dimensions: { {k: (v.passed, v.score) for k, v in result.dimensions.items()} }"
        )

    def test_traceability_score_is_one_when_all_claimable_fields_have_citations(self, gate):
        result = gate.evaluate(self._make_null_concerns_input())
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        assert trace.score == 1.0, f"Expected score 1.0; got {trace.score}"

    def test_null_field_not_in_untraced_flags(self, gate):
        """emergency_concerns must not appear in untraced_field flags."""
        result = gate.evaluate(self._make_null_concerns_input())
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        untraced = [f for f in trace.flags if f.startswith("untraced_field:")]
        assert all("emergency_concerns" not in f for f in untraced), (
            f"emergency_concerns=None must not appear in untraced flags; got {trace.flags}"
        )

    def test_non_none_field_without_citation_still_fails(self, gate):
        """A field with a real extracted_value but no valid citation MUST still count as untraced."""
        evidence = [
            LLMEvidence(
                field_name="symptom_update",
                extracted_value="some claim",
                citations=["this text does not exist in the transcript at all"],
                confidence=0.90,
            ),
            LLMEvidence(
                field_name="emergency_concerns",
                extracted_value=None,
                citations=[],
                confidence=0.00,
            ),
        ]
        inp = ArbitrationInput(
            call_id="CALL-NULL-TRACE-FAIL",
            patient_id="PAT-NULL-TRACE",
            transcript=NULL_CONCERNS_TRANSCRIPT,
            llm_evidence=evidence,
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        untraced = [f for f in trace.flags if "symptom_update" in f]
        assert len(untraced) >= 1, (
            f"Non-None field with fabricated citation must be untraced; got {trace.flags}"
        )

    def test_empty_string_extracted_value_also_exempt(self, gate):
        """extracted_value='' (empty string) is also no positive claim — must be exempt."""
        evidence = [
            LLMEvidence(
                field_name="symptom_update",
                extracted_value="improved",
                citations=["Pain is about 3 out of 10"],
                confidence=0.92,
            ),
            LLMEvidence(
                field_name="emergency_concerns",
                extracted_value="",          # Empty string — same as None semantically.
                citations=[],
                confidence=0.00,
            ),
        ]
        inp = ArbitrationInput(
            call_id="CALL-EMPTY-STR",
            patient_id="PAT-NULL-TRACE",
            transcript=NULL_CONCERNS_TRANSCRIPT,
            llm_evidence=evidence,
            call_signals=NORMAL_SIGNALS,
        )
        from arbitration.dimensions.traceability import TraceabilityDimension
        dim = TraceabilityDimension()
        dim_result = dim.check(inp)
        assert dim_result.passed, (
            f"Empty-string extracted_value must be exempt from traceability; "
            f"score={dim_result.score}, flags={dim_result.flags}"
        )

    def test_all_none_evidence_is_fully_traceable(self, gate):
        """When every field is extracted_value=None, ratio defaults to 1.0."""
        evidence = [
            LLMEvidence(
                field_name=f"field_{i}",
                extracted_value=None,
                citations=[],
                confidence=0.00,
            )
            for i in range(3)
        ]
        inp = ArbitrationInput(
            call_id="CALL-ALL-NONE",
            patient_id="PAT-ALL-NONE",
            transcript=NULL_CONCERNS_TRANSCRIPT,
            llm_evidence=evidence,
            call_signals=NORMAL_SIGNALS,
        )
        # Only traceability dimension is checked here; other dims may fail.
        from arbitration.dimensions.traceability import TraceabilityDimension
        dim = TraceabilityDimension()
        dim_result = dim.check(inp)
        assert dim_result.passed, (
            f"All-None evidence must be trivially traceable; score={dim_result.score}"
        )
        assert dim_result.score == 1.0
