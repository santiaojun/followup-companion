"""
Dimension 3 – Out-of-Distribution (OOD) Detection

Checks that the call actually resembles a medical follow-up conversation,
not a wrong-number call, transcription failure, or completely off-topic session.

Pass criteria: no hard OOD signal AND medical keyword density >= minimum.

Why rules instead of embeddings: deterministic, auditable, no latency.
Embedding-based OOD can be layered on top later without changing this interface.
"""
import re

from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# Minimum word count for a transcript to be considered a real call.
MIN_TRANSCRIPT_WORDS: int = 40

# At least this many distinct medical keywords must appear.
MIN_MEDICAL_KEYWORD_HITS: int = 2

# Keywords that strongly suggest a genuine medical follow-up conversation.
MEDICAL_KEYWORDS: list[str] = [
    "symptom", "medication", "medicine", "pain", "doctor", "physician",
    "appointment", "health", "blood pressure", "glucose", "sugar",
    "surgery", "recovery", "nurse", "hospital", "clinic", "prescription",
    "treatment", "follow-up", "follow up", "side effect", "dose", "tablet",
    "injection", "wound", "incision", "healing", "fever", "nausea",
    "dizziness", "shortness of breath", "swelling", "discharge",
]

# Patterns that indicate the call was not a valid medical follow-up.
# Each entry is (flag_tag, compiled_regex).
OOD_HARD_SIGNALS: list[tuple[str, re.Pattern]] = [
    (
        "wrong_number",
        re.compile(
            r"\b(wrong number|I don'?t know you|who (is|are) (this|you)|"
            r"I have no appointments?|you have the wrong (person|number))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "transcription_failure",
        re.compile(
            r"\[TRANSCRIPTION[_\s]?(ERROR|FAILED|UNAVAILABLE)\]",
            re.IGNORECASE,
        ),
    ),
    (
        "non_consent",
        re.compile(
            r"\b(do not call|stop calling|remove (me|my number)|unsubscribe)\b",
            re.IGNORECASE,
        ),
    ),
]


class OODDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.OOD

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        flags: list[str] = []
        transcript = inp.transcript

        # --- Hard check 1: minimum length ---
        word_count = len(transcript.split())
        if word_count < MIN_TRANSCRIPT_WORDS:
            flags.append(f"transcript_too_short:{word_count}_words")

        # --- Hard check 2: explicit OOD signals ---
        for tag, pattern in OOD_HARD_SIGNALS:
            if pattern.search(transcript):
                flags.append(f"ood_signal:{tag}")

        # --- Soft check: medical keyword density ---
        transcript_lower = transcript.lower()
        keyword_hits = sum(
            1 for kw in MEDICAL_KEYWORDS if kw in transcript_lower
        )
        keyword_score = min(keyword_hits / max(MIN_MEDICAL_KEYWORD_HITS, 1), 1.0)

        if keyword_hits < MIN_MEDICAL_KEYWORD_HITS:
            flags.append(f"low_medical_density:{keyword_hits}_keywords")

        # Aggregate score: penalise hard failures heavily
        hard_failure_count = sum(
            1 for f in flags if not f.startswith("low_medical_density")
        )
        score = max(0.0, keyword_score - 0.5 * hard_failure_count)

        passed = len(flags) == 0

        reason = (
            f"Medical keyword hits: {keyword_hits}. "
            + (f"OOD flags: {flags}." if flags else "No OOD signals detected.")
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=score,
            flags=flags,
            reason=reason,
        )
