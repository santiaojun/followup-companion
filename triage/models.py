"""
Data models for the triage module.

Key design (Second Voice pattern):
  When a claim is ambiguous – either because evidence was insufficient or because
  two candidate interpretations are too close in confidence to pick automatically –
  the system must NOT commit to a single value.  Instead, it surfaces the full
  candidate list to the human reviewer so they can select or edit the correct one.

  This mirrors Second Voice: the AI proposes candidates, a human makes the final
  choice.  The AI never silently breaks a tie on behalf of the patient.

Relationship to arbitration/models.py:
  - LLMEvidence  →  feeds into ArbitrationGate (quality checks)
  - ExtractedClaim →  feeds into TriageQueue (HITL routing)
  Both are produced by the LLM extraction phase; they carry overlapping data but
  serve different consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from arbitration.models import Action

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# If the top-2 candidates' confidences differ by less than this, the claim is
# "too close to call" and must go to HITL instead of being auto-selected.
CANDIDATE_CLOSENESS_THRESHOLD: float = 0.15

# Valid values for ExtractedClaim.evidence_source.
EVIDENCE_SOURCES = frozenset({"direct_quote", "inferred", "insufficient"})


# ---------------------------------------------------------------------------
# ExtractedClaim
# ---------------------------------------------------------------------------

@dataclass
class ExtractedClaim:
    """
    One field extracted by the LLM from a call transcript.

    When the extraction is unambiguous, `candidates` is None and
    `extracted_value` holds the single result.

    When the LLM found multiple plausible interpretations, `candidates` holds
    all of them (sorted descending by confidence) and `extracted_value` holds
    the top pick – but the system must not use that pick automatically if
    `needs_hitl` is True.

    Schema:
        candidates: list[dict] | None
            Each entry: {"text": str, "confidence": float}
            Confidence values are in [0.0, 1.0].
    """
    field_name: str
    evidence_source: str           # "direct_quote" | "inferred" | "insufficient"
    extracted_value: Any           # Top-ranked single value (may be None)
    candidates: list[dict] | None  # Full candidate list, or None if unambiguous
    citations: list[str]           # Verbatim transcript quotes supporting this claim
    confidence: float              # Confidence in extracted_value, in [0.0, 1.0]
    reasoning: str = ""            # LLM chain-of-thought (not stored in patient profile)

    # Set by the extractor when the patient was heard clearly but said
    # something too general to act on - "my arm hurts" with no site,
    # character or duration.
    #
    # This is a different problem from low `confidence`, and it has a
    # different fix. Low confidence means we are unsure of the words and
    # need to re-confirm them (agents/confirmation.py). needs_clarification
    # means the words are certain and the clinical content is thin, so the
    # agent asks the follow-up questions for that symptom family
    # (agents/symptom_probe.py). The two are independent: a claim can be
    # confident and still need clarification, which is exactly the case
    # the tiered confirmation ladder cannot help with.
    needs_clarification: bool = False

    # ------------------------------------------------------------------
    # HITL routing logic
    # ------------------------------------------------------------------

    @property
    def needs_hitl(self) -> bool:
        """
        True when this claim must be sent to a human as a full candidate list
        rather than being auto-accepted.

        Triggers:
          1. evidence_source == "insufficient": the transcript simply did not
             contain enough information to make any determination.
          2. Top-2 candidates are within CANDIDATE_CLOSENESS_THRESHOLD: the LLM
             itself is not confident enough to break the tie.
        """
        if self.evidence_source == "insufficient":
            return True

        if self.candidates and len(self.candidates) >= 2:
            ranked = sorted(
                self.candidates,
                key=lambda c: float(c.get("confidence", 0.0)),
                reverse=True,
            )
            gap = ranked[0]["confidence"] - ranked[1]["confidence"]
            if gap < CANDIDATE_CLOSENESS_THRESHOLD:
                return True

        return False

    @property
    def hitl_candidates(self) -> list[dict]:
        """
        Returns the candidate list to show to the human reviewer.

        If `candidates` is populated, returns those.
        Falls back to a single-entry list built from extracted_value so the
        reviewer always gets at least one option to confirm or edit.
        """
        if self.candidates:
            return sorted(
                self.candidates,
                key=lambda c: float(c.get("confidence", 0.0)),
                reverse=True,
            )
        return [{"text": str(self.extracted_value), "confidence": self.confidence}]


# ---------------------------------------------------------------------------
# HITL review item (one claim → human interface)
# ---------------------------------------------------------------------------

@dataclass
class HITLReviewItem:
    """
    A single claim surfaced for human selection in the review interface.

    The UI should render all `candidates` and let the reviewer pick or
    free-edit, never commit a single AI choice silently.
    """
    field_name: str
    candidates: list[dict]     # [{"text": str, "confidence": float}, ...]  descending
    evidence_source: str
    citations: list[str]       # Supporting quotes so the reviewer can cross-check
    hitl_reason: str           # Machine-readable tag: "insufficient_evidence" | "close_candidates"
    note: str                  # Human-readable explanation shown in the UI


# ---------------------------------------------------------------------------
# Triage packet (full package sent to the review queue)
# ---------------------------------------------------------------------------

@dataclass
class TriagePacket:
    """
    Complete package submitted to the human review queue.

    `hitl_items` contains claims where the human must choose from candidates.
    `auto_accepted_claims` contains claims that passed the confidence gap check
    and can be written to the patient profile without human intervention.
    """
    call_id: str
    patient_id: str
    priority: int              # 1 = risk (urgent), 2 = ambiguous (normal)
    action: Action             # From ArbitrationResult
    risk_flags: list[str]      # From ArbitrationResult (empty unless TRIAGE_RISK)
    hitl_items: list[HITLReviewItem]
    auto_accepted_claims: list[ExtractedClaim]
    arbitration_rejection_reason: Optional[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def requires_human_selection(self) -> bool:
        """True when at least one claim needs a human to pick from candidates."""
        return len(self.hitl_items) > 0
