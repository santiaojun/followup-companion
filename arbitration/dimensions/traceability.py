"""
Dimension 2 – Evidence Traceability

Every LLMEvidence item must cite at least one verbatim quote from the transcript,
and that quote must actually appear in the transcript (substring match, case-insensitive).

Pass criteria: traceability ratio >= TRACEABILITY_THRESHOLD.

Why this matters: prevents the LLM from "hallucinating" citations that don't exist
in the actual call, which would make the extracted profile unauditable.
"""
import re

from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# Minimum characters for a citation to be considered substantive (not a trivial fragment).
MIN_CITATION_LENGTH: int = 10

# Trailing characters stripped from a citation before the substring match.
#
# When a model quotes a clause out of the middle of a sentence it tends to
# terminate the quote with a full stop the transcript does not have:
#
#   citation   "I have absolutely no thoughts of ending my life."
#   transcript "...no thoughts of ending my life and I genuinely meant it."
#
# The quote is genuinely verbatim; only the appended punctuation differs, so
# an exact match rejected a citation that really does exist (4/10 runs on
# SC-025). Stripping trailing punctuation removes that false failure without
# weakening the guarantee this dimension exists for: a hallucinated citation
# still fails, because the remaining text still has to appear verbatim.
# Internal punctuation is deliberately NOT normalised - that would start
# accepting paraphrase.
_TRAILING_PUNCTUATION = ' \t\n\r.,;:!?"\'’”'


def _normalise_citation(citation: str) -> str:
    """Lower-case and strip surrounding whitespace / terminal punctuation."""
    return citation.lower().strip(_TRAILING_PUNCTUATION)

# Fraction of evidence items that must have at least one valid transcript citation.
TRACEABILITY_THRESHOLD: float = 0.90


class TraceabilityDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.TRACEABILITY

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        if not inp.llm_evidence:
            # No evidence at all – treat as complete failure
            return DimensionResult(
                name=self.name,
                passed=False,
                score=0.0,
                flags=["no_evidence_provided"],
                reason="LLM produced zero evidence items.",
            )

        transcript_lower = inp.transcript.lower()
        traceable_items = 0
        traceable_count = 0
        untraced_fields: list[str] = []

        for ev in inp.llm_evidence:
            if ev.extracted_value is None or str(ev.extracted_value).strip() == "":
                continue  # No positive claim made (None or empty string); nothing to trace.
            traceable_items += 1
            if self._has_valid_citation(ev.citations, transcript_lower):
                traceable_count += 1
            else:
                untraced_fields.append(ev.field_name)

        if traceable_items == 0:
            ratio = 1.0  # All fields were None — no claims to trace; trivially traceable.
        else:
            ratio = traceable_count / traceable_items

        passed = ratio >= TRACEABILITY_THRESHOLD

        flags = [f"untraced_field:{f}" for f in untraced_fields]
        reason = (
            f"Traceability {ratio:.0%} ({traceable_count}/{traceable_items} claimable items "
            f"have valid transcript citations). "
            + (f"Untraced: {untraced_fields}." if untraced_fields else "")
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=ratio,
            flags=flags,
            reason=reason,
        )

    @staticmethod
    def _has_valid_citation(citations: list[str], transcript_lower: str) -> bool:
        """Return True if at least one citation is substantive and found in transcript."""
        for citation in citations:
            # Substantiveness is judged on the citation as given. Measuring
            # the stripped form instead silently raised the bar by one
            # character and rejected "Yes I did." (10 raw, 9 stripped) - a
            # perfectly good verbatim quote, which broke SC-026 5/5.
            if len(citation) < MIN_CITATION_LENGTH:
                continue
            normalised = _normalise_citation(citation)
            # Punctuation-only citations normalise to "", and "" is a
            # substring of everything, so require real content.
            if not any(ch.isalnum() for ch in normalised):
                continue
            if normalised in transcript_lower:
                return True
        return False
