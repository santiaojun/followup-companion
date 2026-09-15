"""
Data models for the pre-call planning layer (agents/).

Where this sits
---------------
eval/extractor.py answers "what did the patient say?" *after* a call.
agents/ answers "what should we ask, and how?" *before and during* one.

Three concerns live here, deliberately kept as plain data + deterministic
rules so the LLM layer never owns decision logic (same contract as
arbitration/: the model proposes text, rules decide control flow).

  1. ConfirmationLevel / ConfirmationDecision
        Tiered re-confirmation when a value was not *heard* reliably.
  2. SymptomCategory / ProbeDimension / SymptomProbePlan
        Follow-up questions when a value *was* heard but is clinically vague.
  3. ProtocolItem / PlannedQuestion / CallPlan
        The pre-call script produced by agents/planner.py.

(1) and (2) are independent mechanisms for two different kinds of
ambiguity, and AmbiguityKind is the fork between them - see
classify_ambiguity() in agents/symptom_probe.py.

Patients are English-speaking, so all patient-facing text in this package
is English.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# Tier 1 - confirmation thresholds
# ---------------------------------------------------------------------------
# The bands are contiguous and cover [0.0, 1.0]:
#
#   [0.85, 1.00]  trust it, say nothing                    -> ConfirmationLevel.NONE
#   [0.60, 0.85)  cheap read-back                          -> READBACK
#   [0.40, 0.60)  read the candidates out, patient picks   -> CHOICE
#   [0.00, 0.40)  first pass: still try CHOICE;
#                 after one failed attempt: spell it out
#                 (critical fields only) or hand to a human
#
# Rationale for the ordering: each level costs the patient more time than the
# last, so we only pay for a level when the cheaper one cannot work.

# At or above this, the value is taken as heard. No confirmation utterance.
CONFIRMATION_ACCEPT_THRESHOLD: float = 0.85

# At or above this (and below accept), a single read-back is enough.
CONFIRMATION_READBACK_THRESHOLD: float = 0.60

# At or above this (and below read-back), offer the candidate list.
CONFIRMATION_CHOICE_THRESHOLD: float = 0.40

# Below CONFIRMATION_CHOICE_THRESHOLD a value is treated as "not really heard".
# One confirmation attempt is still spent; after that the field is either
# spelled out (critical) or handed to a human. Never re-asked indefinitely.

# Hard cap on confirmation attempts per field, regardless of level.
# This is the anti-loop guarantee: a patient is never asked about the same
# field more than this many times in one call.
MAX_CONFIRMATION_ATTEMPTS: int = 2

# Offering a choice needs at least this many distinct candidates to read out.
MIN_CANDIDATES_FOR_CHOICE: int = 2


# ---------------------------------------------------------------------------
# Tier 1 - critical fields
# ---------------------------------------------------------------------------
# "Critical" = getting it wrong is both likely (proper nouns survive ASR
# badly) and consequential (wrong drug, wrong patient). Only these fields
# earn the most expensive confirmation level, letter-by-letter spelling.

CRITICAL_FIELDS: frozenset[str] = frozenset({
    "patient_name",
    "medication_name",
    "medication_adherence",
    "medication_dosage",
    "allergy",
    "emergency_concerns",
})

# Substring fallback so a field name we have not enumerated ("new_medication",
# "caregiver_name") is still treated as critical rather than silently demoted.
CRITICAL_FIELD_KEYWORDS: tuple[str, ...] = (
    "name", "medication", "drug", "dose", "dosage", "allerg",
)

# How a field is referred to out loud, when a confirmation utterance has to
# name it. Phrased to drop into a sentence: "... about your medication ...".
FIELD_LABELS: dict[str, str] = {
    "patient_name": "your name",
    "medication_name": "the name of the medication",
    "medication_adherence": "your medication",
    "medication_dosage": "the dose",
    "allergy": "your allergies",
    "symptom_update": "your symptoms",
    "pain_level": "your pain level",
    "appointment_compliance": "your appointment",
    "emergency_concerns": "what you just mentioned",
}


class ConfirmationLevel(IntEnum):
    """
    How hard we work to confirm one value. Ordered, and escalation is
    monotonic: a field never drops back to a cheaper level, which is what
    makes the ladder terminate.
    """
    NONE = 0      # Confidence is high enough; nothing is said.
    READBACK = 1  # "You said X, is that correct?"
    CHOICE = 2    # "Did you say stomach pain, or chest pain?"
    SPELL = 3     # Letter by letter: "M as in Mike, E as in Echo, ...".


class ConfirmationAction(str, Enum):
    """What the caller should do with a ConfirmationDecision."""
    ACCEPT = "accept"                # Use the value as-is.
    CONFIRM = "confirm"              # Speak `utterance`, then re-score.
    MANUAL_REVIEW = "manual_review"  # Stop asking; route to triage/ HITL.


@dataclass
class ConfirmationDecision:
    """
    One verdict from the confirmation ladder.

    `utterance` is empty for ACCEPT and holds a short closing line for
    MANUAL_REVIEW (the patient is told it will be followed up, not
    interrogated further).
    """
    field_name: str
    action: ConfirmationAction
    level: ConfirmationLevel
    utterance: str
    reason: str                                   # Machine-readable tag.
    note: str = ""                                # Human-readable, for the audit log.
    candidates_offered: list[dict] = field(default_factory=list)
    attempt_index: int = 0                        # 0 = first time this field is handled.
    is_critical: bool = False

    @property
    def is_terminal(self) -> bool:
        """True when no further confirmation attempt will be made."""
        return self.action in (
            ConfirmationAction.ACCEPT,
            ConfirmationAction.MANUAL_REVIEW,
        )

    @property
    def needs_hitl(self) -> bool:
        """True when this field must be surfaced to a human reviewer."""
        return self.action is ConfirmationAction.MANUAL_REVIEW


# ---------------------------------------------------------------------------
# Tier 2 - symptom detail probing
# ---------------------------------------------------------------------------

# Cap on probe questions per symptom claim. A follow-up call is not a clinic
# intake; three targeted questions is the practical ceiling before the
# patient starts giving shorter and shorter answers.
MAX_PROBE_QUESTIONS: int = 3

# Fields whose value is a free-text symptom description, i.e. the ones that
# can be "heard perfectly but still too vague to act on".
SYMPTOM_FIELDS: frozenset[str] = frozenset({
    "symptom_update",
    "pain_level",
})


class SymptomCategory(str, Enum):
    """Symptom families that have their own probe checklist."""
    PAIN = "pain"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    WOUND = "wound"
    UNKNOWN = "unknown"


class AmbiguityKind(str, Enum):
    """
    The fork that keeps the two failure modes apart.

    TRANSCRIPTION_UNCLEAR
        We are not sure *what words* were said. Fix with the confirmation
        ladder (agents/confirmation.py). Probing for detail here is useless -
        you cannot ask "where exactly does it hurt" about a sentence you
        did not hear.

    UNDERSPECIFIED
        The words are certain, the clinical content is not. "My arm hurts"
        has no transcription ambiguity at all; it is simply missing the
        site, the character and the duration. Fix with a symptom probe
        (agents/symptom_probe.py).

    NONE
        Nothing further needed.
    """
    NONE = "none"
    TRANSCRIPTION_UNCLEAR = "transcription_unclear"
    UNDERSPECIFIED = "underspecified"


@dataclass(frozen=True)
class ProbeDimension:
    """
    One clinically meaningful attribute of a symptom.

    `present_keywords` is how we decide the patient already covered it, so
    we do not re-ask. Each keyword is matched case-insensitively at a word
    boundary, and matches prefixes - so "constipat" covers "constipation"
    and "constipated", while "gas" would not match "gasping".

    `present_pattern` is an optional regex for the things keywords cannot
    express - "6 out of 10", "three days", "twice a day" - checked in
    addition to the keywords (either one matching counts as covered).
    """
    key: str
    question: str
    required: bool = True                  # False = asked only if budget remains.
    present_keywords: tuple[str, ...] = ()
    present_pattern: str = ""


@dataclass
class SymptomProbePlan:
    """
    Probe questions generated for one incomplete symptom claim.

    `questions` holds only the *required* gaps, capped at MAX_PROBE_QUESTIONS.
    `optional_questions` holds lower-value gaps that fit in whatever budget
    is left over, for a caller that has room for them. `missing` records
    every gap either way, including ones the cap excluded, so the audit
    trail shows the budget decision instead of hiding it.
    """
    field_name: str
    raw_text: str
    category: SymptomCategory
    covered: list[str] = field(default_factory=list)    # Dimension keys already answered.
    missing: list[str] = field(default_factory=list)    # Dimension keys still open.
    questions: list[str] = field(default_factory=list)  # Required gaps, capped.
    optional_questions: list[str] = field(default_factory=list)  # Fills spare budget.

    @property
    def has_questions(self) -> bool:
        return bool(self.questions)

    @property
    def all_questions(self) -> list[str]:
        """Required gaps followed by the optional ones that fit the budget."""
        return self.questions + self.optional_questions


# ---------------------------------------------------------------------------
# Tier 3 - call plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolItem:
    """
    One item the standard follow-up protocol requires us to cover.

    These are non-negotiable: the planner may reword or reorder them, but
    every mandatory item must appear in the resulting CallPlan. That
    invariant is enforced in code, not left to the model.
    """
    field_name: str
    label: str          # Short human-readable name, used in generated text.
    intent: str         # What a clinician needs out of this item.
    mandatory: bool = True


# Mirrors REQUIRED_FIELDS in eval/real_extractor.py. Duplicated rather than
# imported because that module imports openai at import time and hard-exits
# when it is missing; agents/ must stay importable without it.
STANDARD_PROTOCOL_ITEMS: tuple[ProtocolItem, ...] = (
    ProtocolItem(
        field_name="symptom_update",
        label="symptom changes",
        intent="New or changed symptoms since the last contact.",
    ),
    ProtocolItem(
        field_name="medication_adherence",
        label="medication adherence",
        intent="Whether the patient is taking medication as prescribed.",
    ),
    ProtocolItem(
        field_name="pain_level",
        label="pain level",
        intent="Current pain rating, ideally numeric 0-10.",
    ),
    ProtocolItem(
        field_name="appointment_compliance",
        label="appointment attendance",
        intent="Whether recent scheduled appointments were attended.",
    ),
    ProtocolItem(
        field_name="emergency_concerns",
        label="urgent concerns",
        intent="Any urgent concern requiring same-day escalation.",
    ),
)


@dataclass
class PlannedQuestion:
    """One question the agent intends to ask, tied back to a protocol item."""
    field_name: str
    question: str
    rationale: str = ""            # Why this wording / why now. For the audit trail.
    priority: int = 2              # 1 = ask first, 3 = ask if time allows.
    from_history: bool = False     # True when derived from a past call, not the protocol.


@dataclass
class CallPlan:
    """
    The complete pre-call script.

    `personalization_used` is the honest record of whether the patient's
    communication profile was trusted. When profile confidence is below
    memory.profile_store.PERSONALIZATION_CONFIDENCE_THRESHOLD this is False
    and the plan uses neutral defaults - see agents/planner.py.
    """
    patient_id: str
    opening: str
    questions: list[PlannedQuestion]
    tone_directives: list[str]
    continuity_topics: list[str] = field(default_factory=list)
    closing: str = ""

    # Provenance
    personalization_used: bool = False
    profile_confidence: float = 0.0
    profile_call_count: int = 0
    history_call_ids: list[str] = field(default_factory=list)
    model: str = ""
    fallback_used: bool = False          # True when the LLM call failed and the
                                         # deterministic protocol script was used.
    token_usage: dict[str, Any] = field(default_factory=dict)

    # Audit trail: why the plan looks the way it does - degradations, and
    # any mandatory item the code had to add back after the model omitted it.
    notes: list[str] = field(default_factory=list)

    @property
    def covered_fields(self) -> set[str]:
        return {q.field_name for q in self.questions}

    def missing_mandatory(
        self,
        protocol: tuple[ProtocolItem, ...] = STANDARD_PROTOCOL_ITEMS,
    ) -> list[str]:
        """Mandatory protocol items absent from this plan (should always be empty)."""
        covered = self.covered_fields
        return [p.field_name for p in protocol if p.mandatory and p.field_name not in covered]
