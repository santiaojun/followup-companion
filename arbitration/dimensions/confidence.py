"""
Dimension 5 – Confidence Threshold

Evaluates the aggregate confidence of LLM extractions.
A single low-confidence required field acts as a soft flag even if the mean is high.

Decision tiers:
  score >= ACCEPT_THRESHOLD    → pass
  score >= AMBIGUOUS_THRESHOLD → fail (routes to TRIAGE_AMBIGUOUS)
  score <  AMBIGUOUS_THRESHOLD → fail (may route to REJECT if compounded)
"""
from arbitration.dimensions.base import BaseDimension
from arbitration.dimensions.completeness import REQUIRED_FIELDS
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

ACCEPT_THRESHOLD: float = 0.70
AMBIGUOUS_THRESHOLD: float = 0.50

# Required fields whose individual confidence drops below this are flagged separately.
MIN_REQUIRED_FIELD_CONFIDENCE: float = 0.40


class ConfidenceDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.CONFIDENCE

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        if not inp.llm_evidence:
            return DimensionResult(
                name=self.name,
                passed=False,
                score=0.0,
                flags=["no_evidence_provided"],
                reason="No LLM evidence to score.",
            )

        confidences = [e.confidence for e in inp.llm_evidence]
        mean_confidence = sum(confidences) / len(confidences)

        flags: list[str] = []

        # Flag individual required fields with dangerously low confidence.
        evidence_map = {e.field_name: e for e in inp.llm_evidence}
        for field in REQUIRED_FIELDS:
            ev = evidence_map.get(field)
            if ev is not None and ev.confidence < MIN_REQUIRED_FIELD_CONFIDENCE:
                flags.append(
                    f"low_confidence_required_field:{field}:{ev.confidence:.2f}"
                )

        passed = mean_confidence >= ACCEPT_THRESHOLD

        if mean_confidence < AMBIGUOUS_THRESHOLD:
            flags.append(f"aggregate_below_ambiguous_threshold:{mean_confidence:.2f}")

        reason = (
            f"Mean confidence: {mean_confidence:.2f} "
            f"(threshold for accept: {ACCEPT_THRESHOLD}). "
            + (f"Low-confidence fields: {flags}." if flags else "All fields adequately confident.")
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=mean_confidence,
            flags=flags,
            reason=reason,
        )
