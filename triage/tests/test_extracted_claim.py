"""
Tests for ExtractedClaim – specifically the needs_hitl property and hitl_candidates.

Cases:
  A. Unambiguous direct quote       → needs_hitl = False
  B. evidence_source = "insufficient" → needs_hitl = True (always)
  C. Close candidates (gap < 0.15)  → needs_hitl = True
  D. Clear winner (gap ≥ 0.15)      → needs_hitl = False
  E. Single candidate               → needs_hitl = False (no tie possible)
  F. hitl_candidates sorting        → always descending by confidence
  G. hitl_candidates fallback       → when candidates is None, builds from extracted_value
"""
import pytest

from triage.models import CANDIDATE_CLOSENESS_THRESHOLD, ExtractedClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim(
    evidence_source: str = "direct_quote",
    extracted_value=None,
    candidates=None,
    confidence: float = 0.90,
    field_name: str = "pain_level",
) -> ExtractedClaim:
    return ExtractedClaim(
        field_name=field_name,
        evidence_source=evidence_source,
        extracted_value=extracted_value or "3/10",
        candidates=candidates,
        citations=["I would say around 3 out of 10."],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# A. Unambiguous direct quote
# ---------------------------------------------------------------------------

class TestUnambiguousClaim:
    def test_needs_hitl_false_for_direct_quote_no_candidates(self):
        claim = _claim(evidence_source="direct_quote", candidates=None)
        assert claim.needs_hitl is False

    def test_needs_hitl_false_for_inferred_no_candidates(self):
        claim = _claim(evidence_source="inferred", candidates=None)
        assert claim.needs_hitl is False

    def test_needs_hitl_false_for_single_candidate(self):
        claim = _claim(
            evidence_source="inferred",
            candidates=[{"text": "adherent", "confidence": 0.88}],
        )
        assert claim.needs_hitl is False


# ---------------------------------------------------------------------------
# B. evidence_source == "insufficient"
# ---------------------------------------------------------------------------

class TestInsufficientEvidence:
    def test_needs_hitl_true_when_insufficient(self):
        claim = _claim(evidence_source="insufficient", confidence=0.10)
        assert claim.needs_hitl is True

    def test_needs_hitl_true_even_with_no_candidates(self):
        """Insufficient evidence with no candidates should still need HITL."""
        claim = _claim(evidence_source="insufficient", candidates=None)
        assert claim.needs_hitl is True

    def test_needs_hitl_true_even_if_single_high_confidence_candidate(self):
        """Evidence source drives HITL, not candidate count alone."""
        claim = _claim(
            evidence_source="insufficient",
            candidates=[{"text": "yes", "confidence": 0.95}],
        )
        assert claim.needs_hitl is True


# ---------------------------------------------------------------------------
# C. Close candidates (gap < CANDIDATE_CLOSENESS_THRESHOLD)
# ---------------------------------------------------------------------------

class TestCloseCandidates:
    def _close_pair(self, gap: float) -> list[dict]:
        """Build two candidates separated by exactly `gap`."""
        return [
            {"text": "yes – adhering", "confidence": 0.70},
            {"text": "partial – occasional misses", "confidence": round(0.70 - gap, 4)},
        ]

    def test_needs_hitl_true_when_gap_is_zero(self):
        claim = _claim(candidates=self._close_pair(0.0))
        assert claim.needs_hitl is True

    def test_needs_hitl_true_when_gap_just_below_threshold(self):
        gap = CANDIDATE_CLOSENESS_THRESHOLD - 0.01
        claim = _claim(candidates=self._close_pair(gap))
        assert claim.needs_hitl is True

    def test_needs_hitl_true_when_gap_equals_threshold_minus_epsilon(self):
        gap = CANDIDATE_CLOSENESS_THRESHOLD - 0.001
        claim = _claim(candidates=self._close_pair(gap))
        assert claim.needs_hitl is True

    def test_needs_hitl_false_when_gap_above_threshold(self):
        """Gap clearly above threshold: top candidate wins, no HITL needed.
        Note: avoid testing gap == threshold exactly — floating-point subtraction
        (0.70 - 0.55) yields ~0.14999… which still satisfies < 0.15.
        Use a gap safely above threshold instead."""
        gap = CANDIDATE_CLOSENESS_THRESHOLD + 0.05  # 0.20, unambiguously above
        claim = _claim(candidates=self._close_pair(gap))
        assert claim.needs_hitl is False

    def test_needs_hitl_false_when_gap_above_threshold(self):
        gap = CANDIDATE_CLOSENESS_THRESHOLD + 0.10
        claim = _claim(candidates=self._close_pair(gap))
        assert claim.needs_hitl is False


# ---------------------------------------------------------------------------
# D. Clear winner
# ---------------------------------------------------------------------------

class TestClearWinner:
    def test_clear_winner_does_not_need_hitl(self):
        claim = _claim(
            evidence_source="direct_quote",
            candidates=[
                {"text": "3/10", "confidence": 0.95},
                {"text": "4/10", "confidence": 0.40},  # gap = 0.55
            ],
        )
        assert claim.needs_hitl is False

    def test_multiple_candidates_one_dominant(self):
        claim = _claim(
            evidence_source="inferred",
            candidates=[
                {"text": "adherent", "confidence": 0.88},
                {"text": "partial", "confidence": 0.50},
                {"text": "non-adherent", "confidence": 0.20},
            ],
        )
        assert claim.needs_hitl is False


# ---------------------------------------------------------------------------
# E. Single candidate
# ---------------------------------------------------------------------------

class TestSingleCandidate:
    def test_single_candidate_direct_quote_no_hitl(self):
        claim = _claim(
            evidence_source="direct_quote",
            candidates=[{"text": "attended", "confidence": 0.93}],
        )
        assert claim.needs_hitl is False

    def test_single_candidate_inferred_no_hitl(self):
        claim = _claim(
            evidence_source="inferred",
            candidates=[{"text": "no emergency concerns", "confidence": 0.82}],
        )
        assert claim.needs_hitl is False


# ---------------------------------------------------------------------------
# F. hitl_candidates sorting
# ---------------------------------------------------------------------------

class TestHITLCandidatesSorting:
    def test_candidates_returned_descending_by_confidence(self):
        claim = _claim(
            candidates=[
                {"text": "partial", "confidence": 0.50},
                {"text": "adherent", "confidence": 0.88},
                {"text": "non-adherent", "confidence": 0.20},
            ]
        )
        result = claim.hitl_candidates
        confidences = [c["confidence"] for c in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_first_candidate_is_highest_confidence(self):
        claim = _claim(
            candidates=[
                {"text": "mild fatigue", "confidence": 0.65},
                {"text": "severe fatigue", "confidence": 0.68},
            ]
        )
        assert claim.hitl_candidates[0]["confidence"] == 0.68


# ---------------------------------------------------------------------------
# G. hitl_candidates fallback when candidates is None
# ---------------------------------------------------------------------------

class TestHITLCandidatesFallback:
    def test_fallback_builds_single_entry_from_extracted_value(self):
        claim = _claim(
            evidence_source="insufficient",
            extracted_value="unknown",
            candidates=None,
            confidence=0.20,
        )
        result = claim.hitl_candidates
        assert len(result) == 1
        assert result[0]["text"] == "unknown"
        assert result[0]["confidence"] == 0.20
