"""
TriageQueue – builds TriagePackets from arbitration results and extracted claims.

Routing rules:
  - TRIAGE_RISK packets: all claims pass through (reviewer is already looking at
    the call for safety reasons; might as well review claim quality too).
  - TRIAGE_AMBIGUOUS packets: claims with needs_hitl=True become HITLReviewItems
    (full candidate list); claims with needs_hitl=False are auto-accepted.
  - ACCEPT packets: should not reach the triage queue, but are handled gracefully
    by creating a packet with no HITL items (purely for logging / audit).

The queue itself is stateless in this implementation.  Persistence (database,
message broker, etc.) is the caller's responsibility.
"""
from __future__ import annotations

from arbitration.models import Action, ArbitrationResult
from triage.models import (
    CANDIDATE_CLOSENESS_THRESHOLD,
    ExtractedClaim,
    HITLReviewItem,
    TriagePacket,
)


def _hitl_reason_tag(claim: ExtractedClaim) -> str:
    """Machine-readable tag explaining why this claim goes to HITL."""
    if claim.evidence_source == "insufficient":
        return "insufficient_evidence"
    return "close_candidates"


def _hitl_note(claim: ExtractedClaim) -> str:
    """Human-readable note shown in the reviewer UI."""
    if claim.evidence_source == "insufficient":
        return (
            f"Field '{claim.field_name}': transcript did not contain enough "
            f"information to make a determination. Please listen to the recording "
            f"and select or enter the correct value."
        )
    # Close-confidence case: show the gap
    if claim.candidates and len(claim.candidates) >= 2:
        ranked = sorted(
            claim.candidates,
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )
        gap = ranked[0]["confidence"] - ranked[1]["confidence"]
        return (
            f"Field '{claim.field_name}': top two candidates are within "
            f"{gap:.0%} confidence of each other "
            f"(threshold: {CANDIDATE_CLOSENESS_THRESHOLD:.0%}). "
            f"Please select the correct interpretation."
        )
    return f"Field '{claim.field_name}': requires human review."


class TriageQueue:
    """
    Converts an ArbitrationResult + list[ExtractedClaim] into a TriagePacket.

    Usage:
        queue = TriageQueue()
        packet = queue.build_packet(arbitration_result, claims)
        if packet.requires_human_selection:
            review_interface.submit(packet)
    """

    def build_packet(
        self,
        arbitration_result: ArbitrationResult,
        claims: list[ExtractedClaim],
    ) -> TriagePacket:
        """
        Build a TriagePacket.

        For TRIAGE_RISK: route ALL claims to HITL (human is reviewing for
        safety anyway; candidate ambiguity is resolved in the same pass).

        For TRIAGE_AMBIGUOUS / ACCEPT: route only claims where needs_hitl=True
        to HITL; auto-accept the rest.
        """
        if arbitration_result.action == Action.TRIAGE_RISK:
            hitl_items = [self._to_hitl_item(c) for c in claims]
            auto_accepted: list[ExtractedClaim] = []
        else:
            hitl_items = [
                self._to_hitl_item(c) for c in claims if c.needs_hitl
            ]
            auto_accepted = [c for c in claims if not c.needs_hitl]

        return TriagePacket(
            call_id=arbitration_result.call_id,
            patient_id=self._patient_id_from(arbitration_result),
            priority=arbitration_result.priority,
            action=arbitration_result.action,
            risk_flags=arbitration_result.risk_flags,
            hitl_items=hitl_items,
            auto_accepted_claims=auto_accepted,
            arbitration_rejection_reason=arbitration_result.rejection_reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_hitl_item(claim: ExtractedClaim) -> HITLReviewItem:
        return HITLReviewItem(
            field_name=claim.field_name,
            candidates=claim.hitl_candidates,   # Always ≥1 entry, sorted desc by confidence
            evidence_source=claim.evidence_source,
            citations=claim.citations,
            hitl_reason=_hitl_reason_tag(claim),
            note=_hitl_note(claim),
        )

    @staticmethod
    def _patient_id_from(result: ArbitrationResult) -> str:
        # ArbitrationResult does not carry patient_id directly; callers should
        # pass it explicitly in a future refactor.  For now, derive from call_id
        # or return empty string so the packet is still usable.
        return ""
