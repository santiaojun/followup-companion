"""
Tests for TriageQueue.build_packet – candidate routing logic.

Cases:
  A. All claims clear → zero HITL items, all auto-accepted
  B. One insufficient-evidence claim → that claim becomes a HITLReviewItem with
     full candidate list, not a single value
  C. Close-confidence claims → become HITL items
  D. TRIAGE_RISK action → ALL claims go to HITL (human reviewing for safety)
  E. Mixed claims → correct split between HITL and auto-accepted
  F. HITLReviewItem content → candidates list is present and sorted
  G. TriagePacket metadata → priority, action, risk_flags propagated correctly
"""
import pytest

from arbitration.models import Action, ArbitrationResult, DimensionName, DimensionResult
from triage.models import CANDIDATE_CLOSENESS_THRESHOLD, ExtractedClaim
from triage.queue import TriageQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arb_result(
    action: Action = Action.TRIAGE_AMBIGUOUS,
    priority: int = 2,
    risk_flags: list | None = None,
    call_id: str = "CALL-TQ-001",
    rejection_reason: str | None = None,
) -> ArbitrationResult:
    """Minimal ArbitrationResult for queue tests (dimensions not needed)."""
    return ArbitrationResult(
        call_id=call_id,
        passed=(action == Action.ACCEPT),
        action=action,
        priority=priority,
        dimensions={},
        risk_flags=risk_flags or [],
        rejection_reason=rejection_reason,
    )


def _clear_claim(field_name: str = "pain_level") -> ExtractedClaim:
    """A claim with a decisive single extraction – should NOT go to HITL."""
    return ExtractedClaim(
        field_name=field_name,
        evidence_source="direct_quote",
        extracted_value="3/10",
        candidates=[
            {"text": "3/10", "confidence": 0.92},
            {"text": "4/10", "confidence": 0.30},  # gap = 0.62
        ],
        citations=["I would say around 3 out of 10."],
        confidence=0.92,
    )


def _insufficient_claim(field_name: str = "medication_adherence") -> ExtractedClaim:
    """A claim where evidence was insufficient – must go to HITL."""
    return ExtractedClaim(
        field_name=field_name,
        evidence_source="insufficient",
        extracted_value=None,
        candidates=[
            {"text": "adherent", "confidence": 0.40},
            {"text": "non-adherent", "confidence": 0.35},
            {"text": "partial", "confidence": 0.25},
        ],
        citations=[],
        confidence=0.40,
    )


def _close_claim(field_name: str = "symptom_update") -> ExtractedClaim:
    """A claim where top-2 candidates are within the closeness threshold."""
    gap = CANDIDATE_CLOSENESS_THRESHOLD - 0.05  # definitely close
    return ExtractedClaim(
        field_name=field_name,
        evidence_source="inferred",
        extracted_value="mild fatigue",
        candidates=[
            {"text": "mild fatigue", "confidence": 0.68},
            {"text": "moderate fatigue", "confidence": round(0.68 - gap, 4)},
        ],
        citations=["I've been feeling kind of tired."],
        confidence=0.68,
    )


@pytest.fixture
def queue() -> TriageQueue:
    return TriageQueue()


# ---------------------------------------------------------------------------
# A. All claims clear → zero HITL items
# ---------------------------------------------------------------------------

class TestAllClaimsClear:
    def test_no_hitl_items(self, queue):
        result = _arb_result()
        claims = [_clear_claim("pain_level"), _clear_claim("symptom_update")]
        packet = queue.build_packet(result, claims)
        assert packet.hitl_items == []

    def test_all_claims_auto_accepted(self, queue):
        result = _arb_result()
        claims = [_clear_claim("pain_level"), _clear_claim("symptom_update")]
        packet = queue.build_packet(result, claims)
        assert len(packet.auto_accepted_claims) == 2

    def test_requires_human_selection_false(self, queue):
        result = _arb_result()
        claims = [_clear_claim()]
        packet = queue.build_packet(result, claims)
        assert packet.requires_human_selection is False


# ---------------------------------------------------------------------------
# B. Insufficient-evidence claim → HITLReviewItem with full candidate list
# ---------------------------------------------------------------------------

class TestInsufficientEvidenceRouting:
    def test_insufficient_claim_becomes_hitl_item(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        assert len(packet.hitl_items) == 1

    def test_insufficient_claim_not_auto_accepted(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        assert len(packet.auto_accepted_claims) == 0

    def test_hitl_item_has_full_candidate_list(self, queue):
        """All three candidates must appear – the system must not pick one silently."""
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        item = packet.hitl_items[0]
        assert len(item.candidates) == 3

    def test_hitl_item_reason_is_insufficient_evidence(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        item = packet.hitl_items[0]
        assert item.hitl_reason == "insufficient_evidence"

    def test_hitl_item_contains_field_name(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim("medication_adherence")])
        assert packet.hitl_items[0].field_name == "medication_adherence"

    def test_requires_human_selection_true(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        assert packet.requires_human_selection is True


# ---------------------------------------------------------------------------
# C. Close-confidence claim → HITL
# ---------------------------------------------------------------------------

class TestCloseCandidateRouting:
    def test_close_claim_becomes_hitl_item(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_close_claim()])
        assert len(packet.hitl_items) == 1

    def test_close_claim_hitl_reason_is_close_candidates(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_close_claim()])
        assert packet.hitl_items[0].hitl_reason == "close_candidates"

    def test_close_claim_candidates_sorted_descending(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_close_claim()])
        confidences = [c["confidence"] for c in packet.hitl_items[0].candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_close_claim_citations_preserved(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_close_claim("symptom_update")])
        item = packet.hitl_items[0]
        assert "I've been feeling kind of tired." in item.citations


# ---------------------------------------------------------------------------
# D. TRIAGE_RISK – ALL claims go to HITL
# ---------------------------------------------------------------------------

class TestTriageRiskAllClaims:
    def test_all_claims_become_hitl_on_risk(self, queue):
        """Even clear claims must be reviewed when a risk signal was detected."""
        result = _arb_result(
            action=Action.TRIAGE_RISK,
            priority=1,
            risk_flags=["critical:suicidal_ideation"],
        )
        claims = [
            _clear_claim("pain_level"),      # would normally be auto-accepted
            _clear_claim("symptom_update"),  # would normally be auto-accepted
            _insufficient_claim(),           # already HITL
        ]
        packet = queue.build_packet(result, claims)
        assert len(packet.hitl_items) == 3
        assert len(packet.auto_accepted_claims) == 0

    def test_risk_flags_propagated(self, queue):
        result = _arb_result(
            action=Action.TRIAGE_RISK,
            risk_flags=["critical:acute_emergency"],
        )
        packet = queue.build_packet(result, [_clear_claim()])
        assert "critical:acute_emergency" in packet.risk_flags

    def test_priority_is_one_on_risk(self, queue):
        result = _arb_result(action=Action.TRIAGE_RISK, priority=1)
        packet = queue.build_packet(result, [_clear_claim()])
        assert packet.priority == 1


# ---------------------------------------------------------------------------
# E. Mixed claims – correct split
# ---------------------------------------------------------------------------

class TestMixedClaimsSplit:
    def test_correct_split_between_hitl_and_auto(self, queue):
        result = _arb_result()
        claims = [
            _clear_claim("pain_level"),          # auto
            _insufficient_claim("medication_adherence"),  # HITL
            _clear_claim("appointment_compliance"),  # auto
            _close_claim("symptom_update"),      # HITL
        ]
        packet = queue.build_packet(result, claims)
        assert len(packet.hitl_items) == 2
        assert len(packet.auto_accepted_claims) == 2

    def test_hitl_field_names_correct(self, queue):
        result = _arb_result()
        claims = [
            _clear_claim("pain_level"),
            _insufficient_claim("medication_adherence"),
            _close_claim("symptom_update"),
        ]
        packet = queue.build_packet(result, claims)
        hitl_fields = {item.field_name for item in packet.hitl_items}
        assert hitl_fields == {"medication_adherence", "symptom_update"}

    def test_auto_accepted_field_names_correct(self, queue):
        result = _arb_result()
        claims = [
            _clear_claim("pain_level"),
            _insufficient_claim("medication_adherence"),
        ]
        packet = queue.build_packet(result, claims)
        auto_fields = {c.field_name for c in packet.auto_accepted_claims}
        assert auto_fields == {"pain_level"}


# ---------------------------------------------------------------------------
# F. HITLReviewItem content validation
# ---------------------------------------------------------------------------

class TestHITLReviewItemContent:
    def test_note_is_non_empty_string(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        assert isinstance(packet.hitl_items[0].note, str)
        assert len(packet.hitl_items[0].note) > 10

    def test_candidates_list_not_empty(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_close_claim()])
        assert len(packet.hitl_items[0].candidates) >= 1

    def test_each_candidate_has_text_and_confidence(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [_insufficient_claim()])
        for candidate in packet.hitl_items[0].candidates:
            assert "text" in candidate
            assert "confidence" in candidate
            assert isinstance(candidate["confidence"], float)


# ---------------------------------------------------------------------------
# G. TriagePacket metadata propagation
# ---------------------------------------------------------------------------

class TestTriagePacketMetadata:
    def test_call_id_propagated(self, queue):
        result = _arb_result(call_id="CALL-META-999")
        packet = queue.build_packet(result, [_clear_claim()])
        assert packet.call_id == "CALL-META-999"

    def test_action_propagated(self, queue):
        result = _arb_result(action=Action.TRIAGE_AMBIGUOUS)
        packet = queue.build_packet(result, [_clear_claim()])
        assert packet.action == Action.TRIAGE_AMBIGUOUS

    def test_rejection_reason_propagated(self, queue):
        result = _arb_result(rejection_reason="Quality gate failed on: ['confidence']")
        packet = queue.build_packet(result, [_clear_claim()])
        assert "confidence" in (packet.arbitration_rejection_reason or "")

    def test_created_at_is_iso_string(self, queue):
        result = _arb_result()
        packet = queue.build_packet(result, [])
        assert "T" in packet.created_at  # ISO-8601 contains 'T' separator
