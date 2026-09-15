"""
Test suite – Transcription quality dimension.

Scenarios:
  A. Clean transcript (no markers)    → dimension passes, call proceeds normally
  B. High ratio of [unintelligible]   → TRIAGE_AMBIGUOUS, early exit
  C. Too many absolute markers        → TRIAGE_AMBIGUOUS, early exit
  D. Low ASR confidence in metadata   → TRIAGE_AMBIGUOUS, early exit
  E. Early-exit behaviour             → transcription failure short-circuits
                                        other dimensions (quality ones still run
                                        but result is TRIAGE_AMBIGUOUS not ACCEPT)
  F. Marker count exactly at boundary → boundary conditions
"""
import pytest

from arbitration import Action, ArbitrationGate
from arbitration.models import ArbitrationInput, DimensionName
from arbitration.tests.conftest import NORMAL_SIGNALS, build_normal_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inp_with_transcript(transcript: str, call_id: str = "CALL-TQ-001",
                               raw_llm_output: dict | None = None) -> ArbitrationInput:
    inp = build_normal_input(call_id)
    inp.transcript = transcript
    inp.raw_llm_output = raw_llm_output or {}
    return inp


@pytest.fixture
def gate() -> ArbitrationGate:
    return ArbitrationGate()


# ---------------------------------------------------------------------------
# A. Clean transcript – dimension passes
# ---------------------------------------------------------------------------

class TestCleanTranscript:
    def test_transcription_quality_passes_on_normal_call(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.passed is True

    def test_no_transcription_quality_flags_on_normal_call(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.flags == []

    def test_score_is_high_on_clean_transcript(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.score >= 0.70


# ---------------------------------------------------------------------------
# B. High ratio of [unintelligible] markers
# ---------------------------------------------------------------------------

# 8 markers in ~40 words = 20% ratio (above the 15% threshold)
HIGH_RATIO_TRANSCRIPT = (
    "Agent: How have you been feeling since the surgery? "
    "[unintelligible] [unintelligible] [unintelligible] "
    "Patient: [unintelligible] pain [unintelligible] medication [unintelligible] "
    "doctor [unintelligible] appointment [unintelligible] okay I guess."
)


class TestHighUnintelligibleRatio:
    def test_action_is_triage_ambiguous(self, gate):
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT, "CALL-TQ-RATIO")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_passed_is_false(self, gate):
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT)
        result = gate.evaluate(inp)
        assert result.passed is False

    def test_transcription_quality_fails(self, gate):
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT)
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.passed is False

    def test_ratio_flag_present(self, gate):
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT)
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        ratio_flags = [f for f in tq.flags if "unintelligible_ratio" in f]
        assert len(ratio_flags) >= 1

    def test_rejection_reason_mentions_transcription(self, gate):
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT)
        result = gate.evaluate(inp)
        assert "transcription" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# C. Too many absolute marker segments (count threshold)
# ---------------------------------------------------------------------------

# 6 markers but spread over a long transcript → ratio is fine, but count > 5
MANY_MARKERS_TRANSCRIPT = (
    "Agent: Good morning. How are you doing today? "
    "Patient: I have been [unintelligible] and also [unintelligible] which concerns me. "
    "Agent: Can you describe [unintelligible] in more detail? "
    "Patient: Yes, it is like [unintelligible] every morning. The [unintelligible] helps "
    "sometimes but [unintelligible] is still an issue. I take my medication every day and "
    "I attended my appointment last Tuesday. Pain level is about 4 out of 10 overall. "
    "I do not have any emergency concerns at this time. My symptoms have been stable. "
    "The follow-up with the doctor went well and I feel my recovery is on track. "
    "No major issues with medication adherence. I would say I am doing fairly well."
)


class TestAbsoluteMarkerCount:
    def test_action_is_triage_ambiguous(self, gate):
        inp = _make_inp_with_transcript(MANY_MARKERS_TRANSCRIPT, "CALL-TQ-COUNT")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_count_flag_present(self, gate):
        inp = _make_inp_with_transcript(MANY_MARKERS_TRANSCRIPT)
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        count_flags = [f for f in tq.flags if "too_many_unintelligible_segments" in f]
        assert len(count_flags) >= 1

    def test_transcription_quality_fails(self, gate):
        inp = _make_inp_with_transcript(MANY_MARKERS_TRANSCRIPT)
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.passed is False


# ---------------------------------------------------------------------------
# D. Low ASR confidence from CALL-E metadata
# ---------------------------------------------------------------------------

class TestLowASRConfidence:
    def _make(self, confidence: float) -> ArbitrationInput:
        inp = build_normal_input("CALL-TQ-CONF")
        # Inject CALL-E metadata into raw_llm_output
        inp.raw_llm_output = {"transcription_confidence": confidence}
        return inp

    def test_action_is_triage_ambiguous_on_low_confidence(self, gate):
        result = gate.evaluate(self._make(0.45))
        assert result.action == Action.TRIAGE_AMBIGUOUS

    def test_low_asr_confidence_flag_present(self, gate):
        result = gate.evaluate(self._make(0.55))
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        conf_flags = [f for f in tq.flags if "low_asr_confidence" in f]
        assert len(conf_flags) >= 1

    def test_passes_when_confidence_above_threshold(self, gate):
        result = gate.evaluate(self._make(0.85))
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.passed is True

    def test_score_reflects_asr_confidence(self, gate):
        """Score should be capped by the raw ASR confidence value."""
        result = gate.evaluate(self._make(0.50))
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        assert tq.score <= 0.50


# ---------------------------------------------------------------------------
# E. Early-exit: transcription failure short-circuits other dimension logic
# ---------------------------------------------------------------------------

class TestEarlyExitBehaviour:
    def test_transcription_failure_beats_would_be_accept(self, gate):
        """
        A call that would otherwise ACCEPT (all evidence is valid) must still
        return TRIAGE_AMBIGUOUS when transcription quality fails.
        """
        inp = build_normal_input("CALL-TQ-EARLYEXIT")
        # Inject bad ASR confidence – everything else is pristine
        inp.raw_llm_output = {"transcription_confidence": 0.30}
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_AMBIGUOUS
        assert result.passed is False

    def test_dimension_key_present_in_results(self, gate):
        """Transcription quality result must always appear in the dimension map."""
        inp = build_normal_input("CALL-TQ-KEY")
        result = gate.evaluate(inp)
        assert DimensionName.TRANSCRIPTION_QUALITY.value in result.dimensions

    def test_no_risk_flags_on_transcription_failure(self, gate):
        """Transcription-quality short-circuit must not fabricate risk flags."""
        inp = _make_inp_with_transcript(HIGH_RATIO_TRANSCRIPT, "CALL-TQ-NORISK")
        result = gate.evaluate(inp)
        assert result.risk_flags == []


# ---------------------------------------------------------------------------
# F. Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_exactly_five_markers_passes_count_check(self, gate):
        """Exactly MAX_UNINTELLIGIBLE_COUNT (5) markers should not trigger the count flag."""
        base = (
            "Agent: Hello, how are you? "
            "Patient: I have been [unintelligible] [unintelligible] [unintelligible] "
            "[unintelligible] [unintelligible] "
            "but overall my symptoms are stable. I take my medication every morning as "
            "prescribed by the doctor. My pain level is about 3 out of 10 today. "
            "I attended my follow-up appointment last week. No emergency concerns."
        )
        inp = _make_inp_with_transcript(base, "CALL-TQ-BOUND5")
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        count_flags = [f for f in tq.flags if "too_many_unintelligible_segments" in f]
        assert count_flags == [], "Exactly 5 markers should not exceed the threshold"

    def test_confidence_exactly_at_threshold_passes(self, gate):
        """ASR confidence exactly at MIN_TRANSCRIPTION_CONFIDENCE (0.70) should pass."""
        inp = build_normal_input("CALL-TQ-BOUND70")
        inp.raw_llm_output = {"transcription_confidence": 0.70}
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        conf_flags = [f for f in tq.flags if "low_asr_confidence" in f]
        assert conf_flags == [], "Confidence == threshold should not flag"

    def test_confidence_just_below_threshold_fails(self, gate):
        """ASR confidence just below 0.70 should trigger the flag."""
        inp = build_normal_input("CALL-TQ-BOUND69")
        inp.raw_llm_output = {"transcription_confidence": 0.69}
        result = gate.evaluate(inp)
        tq = result.dimensions[DimensionName.TRANSCRIPTION_QUALITY.value]
        conf_flags = [f for f in tq.flags if "low_asr_confidence" in f]
        assert len(conf_flags) >= 1
