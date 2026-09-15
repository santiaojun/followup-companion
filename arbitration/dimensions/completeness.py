"""
Dimension 1 – Completeness

Checks that the LLM has provided evidence for all required follow-up fields.
A field is "covered" when its LLMEvidence confidence meets the minimum threshold.

Pass criteria: coverage ratio >= COMPLETENESS_THRESHOLD.
"""
from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# Fields the LLM must extract for every follow-up call.
# Keys must match LLMEvidence.field_name exactly.
REQUIRED_FIELDS: list[str] = [
    "symptom_update",       # Any new or changed symptoms since last call
    "medication_adherence", # Is patient taking medications as prescribed?
    "pain_level",           # Numeric or descriptive pain rating
    "appointment_compliance", # Did patient attend recent scheduled visits?
    "emergency_concerns",   # Any acute concerns requiring immediate attention?
]

# A field is "covered" only if LLM confidence meets this floor.
MIN_FIELD_CONFIDENCE: float = 0.30

# Fraction of REQUIRED_FIELDS that must be covered to pass.
COMPLETENESS_THRESHOLD: float = 0.80


class CompletenessDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.COMPLETENESS

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        # Index evidence by field name (last entry wins if duplicate)
        evidence_map = {e.field_name: e for e in inp.llm_evidence}

        covered: list[str] = []
        missing: list[str] = []

        for field in REQUIRED_FIELDS:
            ev = evidence_map.get(field)
            if ev is not None and ev.confidence >= MIN_FIELD_CONFIDENCE:
                covered.append(field)
            else:
                missing.append(field)

        coverage = len(covered) / len(REQUIRED_FIELDS)
        passed = coverage >= COMPLETENESS_THRESHOLD

        flags = [f"missing_field:{f}" for f in missing]
        reason = (
            f"Coverage {coverage:.0%} ({len(covered)}/{len(REQUIRED_FIELDS)} fields). "
            + (f"Missing: {missing}." if missing else "All required fields present.")
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=coverage,
            flags=flags,
            reason=reason,
        )
