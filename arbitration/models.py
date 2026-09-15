"""
Data models for the arbitration gate.

Design contract:
  - LLM produces LLMEvidence (with citations and confidence).
  - ArbitrationGate applies deterministic rules and emits ArbitrationResult.
  - No LLM calls happen inside this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Action(str, Enum):
    """Possible outcomes from the arbitration gate."""
    ACCEPT = "accept"
    TRIAGE_RISK = "triage_risk"           # Escalate immediately – risk signal found
    TRIAGE_AMBIGUOUS = "triage_ambiguous" # Human review – low confidence / incomplete
    REJECT = "reject"                     # Hard stop – PII leak or unrecoverable error


class DimensionName(str, Enum):
    """The seven gate dimensions, in evaluation order."""
    COMPLETENESS = "completeness"
    TRACEABILITY = "traceability"
    OOD = "ood"
    RISK_SIGNAL = "risk_signal"
    CONFIDENCE = "confidence"
    PII_BOUNDARY = "pii_boundary"
    TRANSCRIPTION_QUALITY = "transcription_quality"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

@dataclass
class LLMEvidence:
    """
    One structured field extracted by the LLM from the call transcript.

    The LLM is responsible for producing these objects.
    Arbitration rules are responsible for validating them.
    """
    field_name: str
    extracted_value: Any          # Structured value (str, int, bool, …)
    citations: list[str]          # Verbatim quotes from transcript supporting this extraction
    confidence: float             # Self-reported LLM confidence in [0.0, 1.0]
    reasoning: str = ""           # Optional chain-of-thought (not stored in patient profile)


@dataclass
class CallSignals:
    """
    Objective acoustic / timing signals recorded by the telephony platform.
    These are hardware-measured and not subject to LLM interpretation.
    """
    avg_response_latency_ms: float   # Mean time patient paused before responding
    pause_frequency: float           # Pauses per minute (threshold: > 2 s counts as pause)
    interruption_count: int          # Times patient cut off the agent mid-utterance
    call_duration_s: float
    speech_rate_wpm: Optional[float] = None  # Patient words per minute (None if unavailable)


@dataclass
class ArbitrationInput:
    """
    Complete package submitted to the arbitration gate after a call concludes.

    Populated by the orchestrator layer, not by the arbitration module itself.
    """
    call_id: str
    patient_id: str
    transcript: str               # Full verbatim transcript (may contain PII – source of truth)
    llm_evidence: list[LLMEvidence]
    call_signals: CallSignals
    raw_llm_output: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    """
    Outcome of a single dimension check.
    Immutable once created; the gate aggregates these into ArbitrationResult.
    """
    name: DimensionName
    passed: bool
    score: float              # Normalised quality score in [0.0, 1.0]
    flags: list[str]          # Machine-readable issue tags (e.g. "missing_field:pain_level")
    reason: str               # Human-readable explanation for the audit trail


@dataclass
class ArbitrationResult:
    """
    Final verdict produced by ArbitrationGate.evaluate().
    This is the single object handed off to triage/ or stored in memory/.
    """
    call_id: str
    passed: bool               # True only when all six dimensions pass
    action: Action
    priority: int              # 1 = highest urgency (risk), 2 = medium, 3 = routine
    dimensions: dict[str, DimensionResult]
    risk_flags: list[str]      # Aggregated risk tags from the risk_signal dimension
    pii_scrubbed: bool = False  # Set True if PII was detected and must be stripped upstream
    rejection_reason: Optional[str] = None
