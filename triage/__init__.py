"""
triage – HITL routing queue for FollowUp Companion.

Public surface:
    from triage import ExtractedClaim, TriageQueue, TriagePacket, HITLReviewItem
"""
from triage.models import (
    CANDIDATE_CLOSENESS_THRESHOLD,
    ExtractedClaim,
    HITLReviewItem,
    TriagePacket,
)
from triage.queue import TriageQueue

__all__ = [
    "CANDIDATE_CLOSENESS_THRESHOLD",
    "ExtractedClaim",
    "HITLReviewItem",
    "TriagePacket",
    "TriageQueue",
]
