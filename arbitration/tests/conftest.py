"""
Shared test fixtures for arbitration unit tests.

Factory helpers build well-formed ArbitrationInput objects.
Individual test modules tweak specific fields to exercise edge cases.
"""
import pytest

from arbitration.models import ArbitrationInput, CallSignals, LLMEvidence

# ---------------------------------------------------------------------------
# A realistic transcript for a routine post-op follow-up call
# ---------------------------------------------------------------------------
NORMAL_TRANSCRIPT = """
Agent: Hello, this is the FollowUp Companion calling for a post-surgery check-in.
  Am I speaking with the patient?
Patient: Yes, this is me.
Agent: Great. How have you been feeling since the surgery? Any new symptoms?
Patient: I've had some mild fatigue, but overall I feel okay. No major symptoms.
Agent: Are you taking your medications as prescribed?
Patient: Yes, I take my medication every morning as instructed by the doctor.
Agent: On a scale of 1 to 10, how would you rate your pain level today?
Patient: I would say around 3 out of 10. It's manageable.
Agent: Did you attend your follow-up appointment at the clinic last week?
Patient: Yes, I went to the appointment on Tuesday.
Agent: Are there any emergency concerns or anything urgent you would like to report?
Patient: No, nothing urgent. Everything seems to be going well with my recovery.
Agent: That's great to hear. Take care and we will follow up again next week.
Patient: Thank you, goodbye.
""".strip()

NORMAL_SIGNALS = CallSignals(
    avg_response_latency_ms=1200.0,
    pause_frequency=0.8,
    interruption_count=0,
    call_duration_s=185.0,
    speech_rate_wpm=120.0,
)

NORMAL_EVIDENCE = [
    LLMEvidence(
        field_name="symptom_update",
        extracted_value="mild fatigue, no major symptoms",
        citations=["I've had some mild fatigue, but overall I feel okay. No major symptoms."],
        confidence=0.92,
        reasoning="Patient explicitly described mild fatigue and denied major symptoms.",
    ),
    LLMEvidence(
        field_name="medication_adherence",
        extracted_value="adherent",
        citations=["Yes, I take my medication every morning as instructed by the doctor."],
        confidence=0.95,
        reasoning="Clear affirmation of daily medication intake.",
    ),
    LLMEvidence(
        field_name="pain_level",
        extracted_value="3/10",
        citations=["I would say around 3 out of 10. It's manageable."],
        confidence=0.97,
        reasoning="Patient gave a numeric rating of 3 out of 10.",
    ),
    LLMEvidence(
        field_name="appointment_compliance",
        extracted_value="attended",
        citations=["Yes, I went to the appointment on Tuesday."],
        confidence=0.93,
        reasoning="Patient confirmed attendance at clinic appointment.",
    ),
    LLMEvidence(
        field_name="emergency_concerns",
        extracted_value="none",
        citations=["No, nothing urgent. Everything seems to be going well with my recovery."],
        confidence=0.90,
        reasoning="Patient explicitly denied any emergency concerns.",
    ),
]


def build_normal_input(call_id: str = "CALL-001") -> ArbitrationInput:
    """Return a fully valid ArbitrationInput representing a routine call."""
    return ArbitrationInput(
        call_id=call_id,
        patient_id="PAT-001",
        transcript=NORMAL_TRANSCRIPT,
        llm_evidence=list(NORMAL_EVIDENCE),  # shallow copy so tests can mutate
        call_signals=NORMAL_SIGNALS,
    )
