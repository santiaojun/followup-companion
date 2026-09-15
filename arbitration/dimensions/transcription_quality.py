"""
Dimension 7 – Transcription Quality

Detects calls where the ASR (Automatic Speech Recognition) output is too poor
to be reliably interpreted, regardless of what the LLM managed to extract from it.

This is distinct from OOD:
  - OOD asks "is this call in the right domain?"
  - Transcription Quality asks "is the text itself trustworthy?"

A bad transcript means every downstream dimension (completeness, traceability,
confidence, OOD) is reasoning from noisy input. We short-circuit them and route
directly to human review so a clinician can listen to the original recording.

Signals checked:
  1. Inline markers: [unintelligible], [inaudible], [unclear], [noise], [crosstalk]
  2. Ratio of marker tokens to total word tokens
  3. Absolute marker count (catches short calls with many bad segments)
  4. Optional metadata: raw_llm_output["transcription_confidence"] from CALL-E

Gate behaviour:
  - Triggers TRIAGE_AMBIGUOUS immediately; no other dimensions are consulted.
  - Priority set in gate.py; positioned at the same tier as PII (early exit).
"""
import re

from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# Regex matching ASR unintelligibility markers (case-insensitive, bracketed).
UNINTELLIGIBLE_RE = re.compile(
    r"\[(unintelligible|inaudible|unclear|noise|crosstalk|inaudible portion|"
    r"indistinct|garbled|redacted audio)\]",
    re.IGNORECASE,
)

# More than this fraction of word tokens being markers → fail.
MAX_UNINTELLIGIBLE_RATIO: float = 0.15

# More than this many absolute markers in any transcript → fail
# (catches short calls with several bad segments even below the ratio threshold).
MAX_UNINTELLIGIBLE_COUNT: int = 5

# Key the CALL-E platform uses to pass transcript-level ASR confidence in
# ArbitrationInput.raw_llm_output.  0.0 = entirely uncertain, 1.0 = perfect.
TRANSCRIPTION_CONFIDENCE_KEY: str = "transcription_confidence"
MIN_TRANSCRIPTION_CONFIDENCE: float = 0.70


class TranscriptionQualityDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.TRANSCRIPTION_QUALITY

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        flags: list[str] = []
        transcript = inp.transcript

        # ── Signal 1: inline unintelligibility markers ────────────────
        markers = UNINTELLIGIBLE_RE.findall(transcript)
        marker_count = len(markers)

        # Word-token count: split on whitespace, count markers as one token each.
        total_words = len(transcript.split())
        unintelligible_ratio = marker_count / total_words if total_words > 0 else 0.0

        if unintelligible_ratio > MAX_UNINTELLIGIBLE_RATIO:
            flags.append(
                f"high_unintelligible_ratio:{unintelligible_ratio:.2%}"
                f"_({marker_count}/{total_words}_tokens)"
            )

        if marker_count > MAX_UNINTELLIGIBLE_COUNT:
            flags.append(f"too_many_unintelligible_segments:{marker_count}")

        # ── Signal 2: CALL-E ASR confidence score (optional metadata) ─
        asr_confidence = inp.raw_llm_output.get(TRANSCRIPTION_CONFIDENCE_KEY)
        if asr_confidence is not None:
            try:
                asr_confidence = float(asr_confidence)
                if asr_confidence < MIN_TRANSCRIPTION_CONFIDENCE:
                    flags.append(
                        f"low_asr_confidence:{asr_confidence:.2f}"
                        f"_(threshold:{MIN_TRANSCRIPTION_CONFIDENCE})"
                    )
            except (TypeError, ValueError):
                flags.append("asr_confidence_unparseable")

        passed = len(flags) == 0

        # Quality score: starts at 1.0, penalised by unintelligible ratio and
        # ASR confidence shortfall. Clamped to [0.0, 1.0].
        score = max(0.0, 1.0 - unintelligible_ratio * 2)
        if asr_confidence is not None and isinstance(asr_confidence, float):
            score = min(score, asr_confidence)

        reason = (
            "Transcription quality acceptable."
            if passed
            else (
                f"Transcription quality too low for automated analysis. "
                f"Flags: {flags}. Human must review original recording."
            )
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=score,
            flags=flags,
            reason=reason,
        )
