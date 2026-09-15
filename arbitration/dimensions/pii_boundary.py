"""
Dimension 6 – PII Boundary

Ensures PII does not leak from the raw transcript into the structured extracted fields
that will be persisted in the patient profile / memory store.

Scope:
  - Scans: LLMEvidence.extracted_value (str), LLMEvidence.reasoning (str),
           raw_llm_output values (str)
  - Does NOT scan: LLMEvidence.citations (verbatim transcript quotes — expected to contain PII)
  - Does NOT scan: ArbitrationInput.transcript (the source — PII lives here legitimately)

On detection:
  - Sets passed=False and pii_scrubbed flag so the gate can issue a hard REJECT.
  - The gate, not this dimension, decides whether to reject or scrub-and-continue.
"""
import re

from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# (pii_type_tag, compiled_pattern)
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "phone",
        re.compile(
            r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
        ),
    ),
    (
        "email",
        re.compile(
            r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
        ),
    ),
    (
        "credit_card",
        # Matches 16-digit card numbers with optional separators
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    (
        "date_of_birth_explicit",
        re.compile(
            r"\b(born on|date of birth|dob)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
]


def _scan_text(text: str) -> list[str]:
    """Return list of pii_type tags found in text."""
    found: list[str] = []
    for tag, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(tag)
    return found


class PIIBoundaryDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.PII_BOUNDARY

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        flags: list[str] = []

        for ev in inp.llm_evidence:
            # Check extracted_value
            if isinstance(ev.extracted_value, str):
                hits = _scan_text(ev.extracted_value)
                for pii_type in hits:
                    flags.append(f"pii_in_extracted_value:{ev.field_name}:{pii_type}")

            # Check reasoning (goes into logs, not profile, but still a boundary violation)
            if ev.reasoning:
                hits = _scan_text(ev.reasoning)
                for pii_type in hits:
                    flags.append(f"pii_in_reasoning:{ev.field_name}:{pii_type}")

        # Check raw LLM output values
        for key, value in inp.raw_llm_output.items():
            if isinstance(value, str):
                hits = _scan_text(value)
                for pii_type in hits:
                    flags.append(f"pii_in_raw_output:{key}:{pii_type}")

        passed = len(flags) == 0
        score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(flags))

        reason = (
            "No PII detected in structured extracted fields."
            if passed
            else (
                f"{len(flags)} PII boundary violation(s) detected. "
                f"Extracted data must not be persisted. Flags: {flags}"
            )
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=score,
            flags=flags,
            reason=reason,
        )
