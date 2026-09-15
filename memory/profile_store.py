"""
memory/profile_store.py – per-patient communication behaviour profile.

WHAT IS STORED
--------------
Only objective, observable communication signals recorded by the telephony
platform during each call:

  avg_response_latency_ms  How long the patient pauses before answering
  pause_frequency          Pauses per minute (> 2 s threshold)
  interruption_count       Times the patient cut off the agent
  call_duration_s          Total call length in seconds
  speech_rate_wpm          Patient words per minute (None if unavailable)

WHAT IS NOT STORED
------------------
No personality labels, inferred emotional states, or psychological trait
assessments.  The distinction matters:

  OK: "avg_response_latency_ms: 2400" (measured fact)
  NOT OK: "anxious communicator" (inference, stigmatising, not auditable)
  NOT OK: "tends to minimise symptoms" (psychological judgement)

This boundary prevents the system from building implicit "personality files"
that could bias clinical interactions or constitute unlawful profiling.

CONFIDENCE AND FALLBACK
-----------------------
Confidence is proportional to call count:

  confidence = min(1.0, call_count / CALLS_TO_FULL_CONFIDENCE)

When confidence < PERSONALIZATION_CONFIDENCE_THRESHOLD the profile's
`use_default_script` flag is True.  The call-planning layer must use
neutral, generic pacing rather than personalising to an unreliable profile.

Threshold values
~~~~~~~~~~~~~~~~
  PERSONALIZATION_CONFIDENCE_THRESHOLD = 0.5  (≥ 5 calls)
  CALLS_TO_FULL_CONFIDENCE              = 10   (1.0 at 10 calls)

So with a single call, confidence = 0.10 → use_default_script = True.
With 10 calls, confidence = 1.00 → use_default_script = False.

USAGE
-----
    store = ProfileStore()                 # in-memory (tests / REPL)
    store = ProfileStore(persist_path=...) # JSON persistence

    from arbitration.models import CallSignals
    signals = CallSignals(avg_response_latency_ms=2400, ...)

    store.update("PAT-001", signals)
    profile = store.get_profile("PAT-001")

    if profile.use_default_script:
        script = NEUTRAL_SCRIPT
    else:
        script = personalise(script, profile)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from arbitration.models import CallSignals

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Denominator for the confidence formula: confidence = min(1, calls / this).
CALLS_TO_FULL_CONFIDENCE: int = 10

# Below this confidence level the profile cannot be trusted and the caller
# must fall back to a neutral, non-personalised interaction script.
PERSONALIZATION_CONFIDENCE_THRESHOLD: float = 0.5   # requires ≥ 5 calls


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class PersonalizationProfile:
    """
    Snapshot of a patient's observed communication patterns.

    All numeric values are averages over the stored calls; None means the
    signal was unavailable in all recorded calls.

    ``use_default_script`` is the primary flag for callers:
      True  → too few calls; use neutral generic pacing and language.
      False → sufficient history; personalisation is reliable enough.
    """
    patient_id: str
    call_count: int
    confidence: float

    # True when confidence < PERSONALIZATION_CONFIDENCE_THRESHOLD.
    # Callers MUST check this before applying any personalisation.
    use_default_script: bool

    # Averaged observed signals.  Present even when use_default_script is True
    # (the values exist but are unreliable until confidence is high enough).
    avg_response_latency_ms: Optional[float] = None
    avg_pause_frequency: Optional[float] = None
    avg_speech_rate_wpm: Optional[float] = None
    avg_interruption_count: Optional[float] = None
    avg_call_duration_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass
class _PatientRecord:
    """Internal storage record per patient (not exposed directly to callers)."""
    call_count: int = 0
    # Running sums for averaging without storing every raw call
    sum_response_latency_ms: float = 0.0
    sum_pause_frequency: float = 0.0
    sum_interruption_count: float = 0.0
    sum_call_duration_s: float = 0.0
    # speech_rate may be None; track separately
    sum_speech_rate_wpm: float = 0.0
    speech_rate_count: int = 0   # number of calls where speech_rate was available

    def add(self, signals: CallSignals) -> None:
        self.call_count += 1
        self.sum_response_latency_ms += signals.avg_response_latency_ms
        self.sum_pause_frequency += signals.pause_frequency
        self.sum_interruption_count += signals.interruption_count
        self.sum_call_duration_s += signals.call_duration_s
        if signals.speech_rate_wpm is not None:
            self.sum_speech_rate_wpm += signals.speech_rate_wpm
            self.speech_rate_count += 1

    def avg_latency(self) -> Optional[float]:
        return self.sum_response_latency_ms / self.call_count if self.call_count else None

    def avg_pause(self) -> Optional[float]:
        return self.sum_pause_frequency / self.call_count if self.call_count else None

    def avg_interruptions(self) -> Optional[float]:
        return self.sum_interruption_count / self.call_count if self.call_count else None

    def avg_duration(self) -> Optional[float]:
        return self.sum_call_duration_s / self.call_count if self.call_count else None

    def avg_speech_rate(self) -> Optional[float]:
        return (
            self.sum_speech_rate_wpm / self.speech_rate_count
            if self.speech_rate_count
            else None
        )

    def to_dict(self) -> dict:
        return {
            "call_count": self.call_count,
            "sum_response_latency_ms": self.sum_response_latency_ms,
            "sum_pause_frequency": self.sum_pause_frequency,
            "sum_interruption_count": self.sum_interruption_count,
            "sum_call_duration_s": self.sum_call_duration_s,
            "sum_speech_rate_wpm": self.sum_speech_rate_wpm,
            "speech_rate_count": self.speech_rate_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_PatientRecord":
        r = cls()
        r.call_count = d.get("call_count", 0)
        r.sum_response_latency_ms = d.get("sum_response_latency_ms", 0.0)
        r.sum_pause_frequency = d.get("sum_pause_frequency", 0.0)
        r.sum_interruption_count = d.get("sum_interruption_count", 0.0)
        r.sum_call_duration_s = d.get("sum_call_duration_s", 0.0)
        r.sum_speech_rate_wpm = d.get("sum_speech_rate_wpm", 0.0)
        r.speech_rate_count = d.get("speech_rate_count", 0)
        return r


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProfileStore:
    """
    Stores and retrieves per-patient communication behaviour profiles.

    Parameters
    ----------
    persist_path : str | Path | None
        Path to a JSON file for durable storage.
        Pass ``None`` for an ephemeral in-memory store (tests / REPL).
        If the file does not exist it is created on the first write.
    """

    def __init__(self, persist_path: Optional[str | Path] = None) -> None:
        self._path: Optional[Path] = Path(persist_path) if persist_path else None
        self._records: dict[str, _PatientRecord] = {}
        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, patient_id: str, signals: CallSignals) -> None:
        """
        Record one call's signals for a patient.

        Calling this N times accumulates N calls into running averages.
        Does not store raw signals — only running sums and counts.
        """
        if patient_id not in self._records:
            self._records[patient_id] = _PatientRecord()
        self._records[patient_id].add(signals)
        if self._path:
            self._save()

    def delete_patient(self, patient_id: str) -> bool:
        """
        Remove all stored data for a patient.

        Returns True if the patient existed, False if not found.
        Intended for GDPR / data-retention workflows.
        """
        if patient_id not in self._records:
            return False
        del self._records[patient_id]
        if self._path:
            self._save()
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_profile(self, patient_id: str) -> PersonalizationProfile:
        """
        Return the current profile for a patient.

        If the patient has no history, returns a zero-call profile with
        ``use_default_script = True`` and all signal averages = None.
        """
        record = self._records.get(patient_id, _PatientRecord())
        confidence = _compute_confidence(record.call_count)
        use_default = confidence < PERSONALIZATION_CONFIDENCE_THRESHOLD

        return PersonalizationProfile(
            patient_id=patient_id,
            call_count=record.call_count,
            confidence=confidence,
            use_default_script=use_default,
            avg_response_latency_ms=record.avg_latency(),
            avg_pause_frequency=record.avg_pause(),
            avg_speech_rate_wpm=record.avg_speech_rate(),
            avg_interruption_count=record.avg_interruptions(),
            avg_call_duration_s=record.avg_duration(),
        )

    def call_count(self, patient_id: str) -> int:
        """Return the number of calls recorded for a patient (0 if unknown)."""
        record = self._records.get(patient_id)
        return record.call_count if record else 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: rec.to_dict() for pid, rec in self._records.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self._path is not None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = {
                pid: _PatientRecord.from_dict(rec)
                for pid, rec in data.items()
            }
        except (json.JSONDecodeError, KeyError):
            self._records = {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_confidence(call_count: int) -> float:
    """
    Linear confidence ramp.

    0 calls  → 0.00
    1 call   → 0.10  (below threshold → use_default_script)
    5 calls  → 0.50  (at threshold boundary)
    10 calls → 1.00  (full confidence)
    """
    return min(1.0, call_count / CALLS_TO_FULL_CONFIDENCE)
