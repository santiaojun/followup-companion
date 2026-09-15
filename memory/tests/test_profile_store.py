"""
Tests for ProfileStore.

Two headline scenarios
----------------------
Scenario A – Low-sample degradation (1 call)
    confidence = 0.10  < PERSONALIZATION_CONFIDENCE_THRESHOLD (0.50)
    use_default_script = True   → call-planner must use neutral pacing

Scenario B – Full-sample personalization (10 calls)
    confidence = 1.00  ≥ PERSONALIZATION_CONFIDENCE_THRESHOLD
    use_default_script = False  → personalization is reliable

Additional coverage
-------------------
- Unknown patient returns zero-call profile with use_default_script = True
- Boundary: exactly 5 calls → confidence == 0.50 (at threshold, NOT below → False)
- Average values computed correctly over multiple calls
- speech_rate_wpm = None handled (not counted in average)
- delete_patient returns True / False, subsequent get_profile resets to zero
- JSON persistence round-trip (persist_path is written and read back)
- call_count() method reflects current record
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from arbitration.models import CallSignals
from memory.profile_store import (
    CALLS_TO_FULL_CONFIDENCE,
    PERSONALIZATION_CONFIDENCE_THRESHOLD,
    ProfileStore,
    _compute_confidence,
)

# ---------------------------------------------------------------------------
# Synthetic CallSignals factories
# ---------------------------------------------------------------------------

def _signals(
    latency_ms: float = 1500.0,
    pause_freq: float = 1.2,
    interruptions: int = 0,
    duration_s: float = 240.0,
    speech_rate_wpm: float | None = 130.0,
) -> CallSignals:
    return CallSignals(
        avg_response_latency_ms=latency_ms,
        pause_frequency=pause_freq,
        interruption_count=interruptions,
        call_duration_s=duration_s,
        speech_rate_wpm=speech_rate_wpm,
    )


# A varied set of signals to test averaging
_CALL_SET = [
    _signals(latency_ms=1000.0, pause_freq=1.0, interruptions=0, duration_s=200.0, speech_rate_wpm=140.0),
    _signals(latency_ms=1200.0, pause_freq=1.5, interruptions=1, duration_s=220.0, speech_rate_wpm=130.0),
    _signals(latency_ms=1400.0, pause_freq=2.0, interruptions=0, duration_s=240.0, speech_rate_wpm=120.0),
    _signals(latency_ms=1600.0, pause_freq=2.5, interruptions=2, duration_s=260.0, speech_rate_wpm=110.0),
    _signals(latency_ms=1800.0, pause_freq=3.0, interruptions=1, duration_s=280.0, speech_rate_wpm=100.0),
    _signals(latency_ms=2000.0, pause_freq=3.5, interruptions=0, duration_s=300.0, speech_rate_wpm=90.0),
    _signals(latency_ms=2200.0, pause_freq=4.0, interruptions=2, duration_s=320.0, speech_rate_wpm=80.0),
    _signals(latency_ms=2400.0, pause_freq=4.5, interruptions=1, duration_s=340.0, speech_rate_wpm=70.0),
    _signals(latency_ms=2600.0, pause_freq=5.0, interruptions=0, duration_s=360.0, speech_rate_wpm=60.0),
    _signals(latency_ms=2800.0, pause_freq=5.5, interruptions=3, duration_s=380.0, speech_rate_wpm=50.0),
]
# Expected averages over all 10 calls:
_EXPECTED_AVG_LATENCY   = sum(c.avg_response_latency_ms for c in _CALL_SET) / 10   # 1900.0
_EXPECTED_AVG_PAUSE     = sum(c.pause_frequency          for c in _CALL_SET) / 10   # 3.25
_EXPECTED_AVG_INTERRUPT = sum(c.interruption_count       for c in _CALL_SET) / 10   # 1.0
_EXPECTED_AVG_DURATION  = sum(c.call_duration_s          for c in _CALL_SET) / 10   # 290.0
_EXPECTED_AVG_SPEECH    = sum(c.speech_rate_wpm          for c in _CALL_SET) / 10   # 95.0


# ---------------------------------------------------------------------------
# Confidence formula unit tests (isolated, no ProfileStore)
# ---------------------------------------------------------------------------

class TestComputeConfidence:

    def test_zero_calls_gives_zero_confidence(self):
        assert _compute_confidence(0) == 0.0

    def test_one_call_gives_0_1(self):
        assert _compute_confidence(1) == pytest.approx(0.10)

    def test_five_calls_gives_0_5(self):
        assert _compute_confidence(5) == pytest.approx(0.50)

    def test_ten_calls_gives_1_0(self):
        assert _compute_confidence(10) == pytest.approx(1.0)

    def test_above_ten_calls_capped_at_1_0(self):
        assert _compute_confidence(20) == pytest.approx(1.0)
        assert _compute_confidence(100) == pytest.approx(1.0)

    def test_formula_is_linear_below_cap(self):
        for n in range(1, CALLS_TO_FULL_CONFIDENCE + 1):
            expected = n / CALLS_TO_FULL_CONFIDENCE
            assert _compute_confidence(n) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Scenario A – Low-sample degradation (1 call)
# ---------------------------------------------------------------------------

class TestLowSampleDegradation:
    """A single call produces a profile that must fall back to neutral scripting."""

    @pytest.fixture
    def store_one_call(self):
        store = ProfileStore()
        store.update("PAT-001", _signals(latency_ms=2400.0, pause_freq=3.0))
        return store

    def test_confidence_below_threshold(self, store_one_call):
        profile = store_one_call.get_profile("PAT-001")
        assert profile.confidence < PERSONALIZATION_CONFIDENCE_THRESHOLD, (
            f"1-call confidence {profile.confidence:.2f} must be below "
            f"threshold {PERSONALIZATION_CONFIDENCE_THRESHOLD}"
        )

    def test_confidence_is_0_1(self, store_one_call):
        profile = store_one_call.get_profile("PAT-001")
        assert profile.confidence == pytest.approx(0.10)

    def test_use_default_script_is_true(self, store_one_call):
        """Call planner MUST use neutral pacing — personalization unreliable."""
        profile = store_one_call.get_profile("PAT-001")
        assert profile.use_default_script is True, (
            "use_default_script must be True with only 1 call recorded"
        )

    def test_call_count_is_one(self, store_one_call):
        profile = store_one_call.get_profile("PAT-001")
        assert profile.call_count == 1

    def test_signal_averages_still_populated(self, store_one_call):
        """
        Even with use_default_script=True the averaged values are still
        present (they're just unreliable, not absent).
        """
        profile = store_one_call.get_profile("PAT-001")
        assert profile.avg_response_latency_ms is not None
        assert profile.avg_pause_frequency is not None

    def test_latency_matches_single_call(self, store_one_call):
        profile = store_one_call.get_profile("PAT-001")
        assert profile.avg_response_latency_ms == pytest.approx(2400.0)

    def test_pause_frequency_matches_single_call(self, store_one_call):
        profile = store_one_call.get_profile("PAT-001")
        assert profile.avg_pause_frequency == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Scenario B – Full-sample personalization (10 calls)
# ---------------------------------------------------------------------------

class TestHighSamplePersonalization:
    """Ten calls cross the threshold — the call-planner should personalise."""

    @pytest.fixture
    def store_ten_calls(self):
        store = ProfileStore()
        for s in _CALL_SET:
            store.update("PAT-002", s)
        return store

    def test_confidence_at_or_above_threshold(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.confidence >= PERSONALIZATION_CONFIDENCE_THRESHOLD, (
            f"10-call confidence {profile.confidence:.2f} must be ≥ threshold "
            f"{PERSONALIZATION_CONFIDENCE_THRESHOLD}"
        )

    def test_confidence_is_1_0(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.confidence == pytest.approx(1.0)

    def test_use_default_script_is_false(self, store_ten_calls):
        """Personalization is now reliable — use_default_script must be False."""
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.use_default_script is False, (
            "use_default_script must be False after 10 calls"
        )

    def test_call_count_is_ten(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.call_count == 10

    def test_avg_latency_correct(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.avg_response_latency_ms == pytest.approx(_EXPECTED_AVG_LATENCY)

    def test_avg_pause_frequency_correct(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.avg_pause_frequency == pytest.approx(_EXPECTED_AVG_PAUSE)

    def test_avg_interruption_count_correct(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.avg_interruption_count == pytest.approx(_EXPECTED_AVG_INTERRUPT)

    def test_avg_call_duration_correct(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.avg_call_duration_s == pytest.approx(_EXPECTED_AVG_DURATION)

    def test_avg_speech_rate_correct(self, store_ten_calls):
        profile = store_ten_calls.get_profile("PAT-002")
        assert profile.avg_speech_rate_wpm == pytest.approx(_EXPECTED_AVG_SPEECH)


# ---------------------------------------------------------------------------
# Unknown patient (0 calls)
# ---------------------------------------------------------------------------

class TestUnknownPatient:

    def test_unknown_patient_has_zero_calls(self):
        store = ProfileStore()
        profile = store.get_profile("PAT-UNKNOWN")
        assert profile.call_count == 0

    def test_unknown_patient_confidence_is_zero(self):
        store = ProfileStore()
        profile = store.get_profile("PAT-UNKNOWN")
        assert profile.confidence == pytest.approx(0.0)

    def test_unknown_patient_uses_default_script(self):
        store = ProfileStore()
        profile = store.get_profile("PAT-UNKNOWN")
        assert profile.use_default_script is True

    def test_unknown_patient_signals_are_none(self):
        store = ProfileStore()
        profile = store.get_profile("PAT-UNKNOWN")
        assert profile.avg_response_latency_ms is None
        assert profile.avg_pause_frequency is None
        assert profile.avg_speech_rate_wpm is None
        assert profile.avg_interruption_count is None
        assert profile.avg_call_duration_s is None

    def test_call_count_method_returns_zero(self):
        store = ProfileStore()
        assert store.call_count("PAT-UNKNOWN") == 0


# ---------------------------------------------------------------------------
# Boundary: exactly 5 calls (confidence == threshold, NOT below)
# ---------------------------------------------------------------------------

class TestBoundaryFiveCalls:
    """
    At exactly 5 calls, confidence == PERSONALIZATION_CONFIDENCE_THRESHOLD (0.50).
    The guard condition is confidence < threshold, so 5 calls → use_default_script = False.
    """

    @pytest.fixture
    def store_five_calls(self):
        store = ProfileStore()
        for _ in range(5):
            store.update("PAT-FIVE", _signals())
        return store

    def test_confidence_equals_threshold(self, store_five_calls):
        profile = store_five_calls.get_profile("PAT-FIVE")
        assert profile.confidence == pytest.approx(PERSONALIZATION_CONFIDENCE_THRESHOLD)

    def test_use_default_script_is_false_at_threshold(self, store_five_calls):
        """
        confidence == threshold is NOT below threshold, so personalization unlocks.
        """
        profile = store_five_calls.get_profile("PAT-FIVE")
        assert profile.use_default_script is False

    def test_four_calls_still_use_default(self):
        store = ProfileStore()
        for _ in range(4):
            store.update("PAT-FOUR", _signals())
        profile = store.get_profile("PAT-FOUR")
        assert profile.use_default_script is True, (
            "4 calls → confidence 0.40 < 0.50 → must still use default script"
        )


# ---------------------------------------------------------------------------
# Optional speech_rate_wpm (None handling)
# ---------------------------------------------------------------------------

class TestSpeechRateNone:

    def test_speech_rate_none_excluded_from_average(self):
        """Calls where speech_rate_wpm is None must not skew the average."""
        store = ProfileStore()
        store.update("PAT-SR", _signals(speech_rate_wpm=None))
        store.update("PAT-SR", _signals(speech_rate_wpm=None))
        profile = store.get_profile("PAT-SR")
        assert profile.avg_speech_rate_wpm is None, (
            "If no call provided speech_rate_wpm, average must be None"
        )

    def test_mixed_none_and_value_averages_only_non_none(self):
        store = ProfileStore()
        store.update("PAT-SR2", _signals(speech_rate_wpm=100.0))
        store.update("PAT-SR2", _signals(speech_rate_wpm=None))   # excluded
        store.update("PAT-SR2", _signals(speech_rate_wpm=200.0))
        profile = store.get_profile("PAT-SR2")
        # Average of 100 and 200 only (None excluded): 150.0
        assert profile.avg_speech_rate_wpm == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# delete_patient
# ---------------------------------------------------------------------------

class TestDeletePatient:

    def test_delete_existing_patient_returns_true(self):
        store = ProfileStore()
        store.update("PAT-DEL", _signals())
        assert store.delete_patient("PAT-DEL") is True

    def test_delete_unknown_patient_returns_false(self):
        store = ProfileStore()
        assert store.delete_patient("PAT-GHOST") is False

    def test_profile_resets_after_delete(self):
        store = ProfileStore()
        for _ in range(8):
            store.update("PAT-RESET", _signals())
        store.delete_patient("PAT-RESET")
        profile = store.get_profile("PAT-RESET")
        assert profile.call_count == 0
        assert profile.use_default_script is True

    def test_delete_does_not_affect_other_patient(self):
        store = ProfileStore()
        for _ in range(10):
            store.update("PAT-KEEP",   _signals())
            store.update("PAT-DELETE", _signals())
        store.delete_patient("PAT-DELETE")
        assert store.call_count("PAT-KEEP") == 10
        assert store.call_count("PAT-DELETE") == 0


# ---------------------------------------------------------------------------
# call_count() method
# ---------------------------------------------------------------------------

class TestCallCountMethod:

    def test_reflects_number_of_updates(self):
        store = ProfileStore()
        for n in range(1, 7):
            store.update("PAT-CNT", _signals())
            assert store.call_count("PAT-CNT") == n

    def test_returns_zero_before_any_update(self):
        store = ProfileStore()
        assert store.call_count("PAT-NEW") == 0


# ---------------------------------------------------------------------------
# JSON persistence round-trip
# ---------------------------------------------------------------------------

class TestJsonPersistence:

    def test_profile_survives_reload(self):
        """Profiles written to disk must be correctly reconstructed on reload."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            store = ProfileStore(persist_path=path)
            for s in _CALL_SET:
                store.update("PAT-PERSIST", s)
            # Reload from the same path
            store2 = ProfileStore(persist_path=path)
            profile = store2.get_profile("PAT-PERSIST")
            assert profile.call_count == 10
            assert profile.confidence == pytest.approx(1.0)
            assert profile.use_default_script is False
            assert profile.avg_response_latency_ms == pytest.approx(_EXPECTED_AVG_LATENCY)
        finally:
            path.unlink(missing_ok=True)

    def test_json_file_is_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            store = ProfileStore(persist_path=path)
            store.update("PAT-JSON", _signals())
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "PAT-JSON" in data
            assert data["PAT-JSON"]["call_count"] == 1
        finally:
            path.unlink(missing_ok=True)

    def test_delete_persists_across_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            store = ProfileStore(persist_path=path)
            store.update("PAT-D1", _signals())
            store.update("PAT-D2", _signals())
            store.delete_patient("PAT-D1")
            store2 = ProfileStore(persist_path=path)
            assert store2.call_count("PAT-D1") == 0
            assert store2.call_count("PAT-D2") == 1
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "does_not_exist.json"
            store = ProfileStore(persist_path=path)
            profile = store.get_profile("PAT-EMPTY")
            assert profile.call_count == 0
