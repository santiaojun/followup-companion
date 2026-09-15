"""
Tests for the tiered confirmation ladder (the "didn't hear it" mechanism).

Three headline scenarios (the ones the design has to get right)
---------------------------------------------------------------
Scenario A - High confidence, no confirmation
    confidence 0.92 -> ACCEPT, empty utterance, nothing said to the patient.

Scenario B - Moderate ambiguity with candidates, choice confirmation
    confidence 0.50 + two candidates -> CHOICE, both options read aloud
    ("Did you say stomach pain, or chest pain?").

Scenario C - Repeated failure ends in human review, not a loop
    confidence stays low across attempts -> the ladder escalates once and
    then stops, returning MANUAL_REVIEW. Critical fields get one spelling
    pass first; non-critical fields do not.

Additional coverage
-------------------
- Band boundaries: 0.85 accepts, 0.60 reads back, 0.40 offers a choice
- CHOICE degrades to READBACK when there is only one distinct candidate
- Duplicate candidate texts are de-duplicated
- More than MAX_CANDIDATES_TO_READ candidates are truncated
- Monotonic escalation: a level is never used twice for the same field
- Every confidence trajectory terminates within a bounded number of turns
- is_critical_field: exact names and substring fallback
- Phonetic spelling for the letter-by-letter tier
- Tracker: only spoken confirmations count; manual review is sticky but
  yields to a later high-confidence reading
"""
from __future__ import annotations

import pytest

from agents.confirmation import (
    MAX_CANDIDATES_TO_READ,
    MAX_CRITICAL_CONFIRMATION_ATTEMPTS,
    MAX_LETTERS_TO_SPELL,
    ConfirmationTracker,
    decide_confirmation,
    field_label,
    is_critical_field,
    spell_out,
)
from agents.models import (
    CONFIRMATION_ACCEPT_THRESHOLD,
    CONFIRMATION_CHOICE_THRESHOLD,
    CONFIRMATION_READBACK_THRESHOLD,
    MAX_CONFIRMATION_ATTEMPTS,
    ConfirmationAction,
    ConfirmationLevel,
)

# A field that is not in CRITICAL_FIELDS and has no critical keyword in it.
PLAIN_FIELD = "symptom_update"
CRITICAL_FIELD = "medication_name"


def _candidates(*pairs: tuple[str, float]) -> list[dict]:
    return [{"text": t, "confidence": c} for t, c in pairs]


TWO_CANDIDATES = _candidates(("stomach pain", 0.52), ("chest pain", 0.48))


# ---------------------------------------------------------------------------
# Scenario A - high confidence does not trigger confirmation
# ---------------------------------------------------------------------------

class TestHighConfidenceAccepts:

    def test_high_confidence_accepts_without_utterance(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.92, value="stomach pain")

        assert d.action is ConfirmationAction.ACCEPT
        assert d.level is ConfirmationLevel.NONE
        assert d.utterance == ""
        assert d.is_terminal
        assert not d.needs_hitl
        assert d.reason == "confidence_above_accept_threshold"

    def test_accept_threshold_is_inclusive(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=CONFIRMATION_ACCEPT_THRESHOLD,
                                value="stomach pain")
        assert d.action is ConfirmationAction.ACCEPT

    def test_just_below_accept_threshold_confirms(self):
        d = decide_confirmation(PLAIN_FIELD,
                                confidence=CONFIRMATION_ACCEPT_THRESHOLD - 0.01,
                                value="stomach pain")
        assert d.action is ConfirmationAction.CONFIRM
        assert d.level is ConfirmationLevel.READBACK

    def test_high_confidence_accepts_even_for_critical_field(self):
        # Criticality raises the ceiling on effort, not the accept threshold.
        d = decide_confirmation(CRITICAL_FIELD, confidence=0.90, value="Metformin")
        assert d.action is ConfirmationAction.ACCEPT

    def test_high_confidence_accepts_mid_ladder(self):
        # Patient repeated it clearly after a failed read-back: that settles it.
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.95,
            value="stomach pain",
            attempted_levels=[ConfirmationLevel.READBACK],
        )
        assert d.action is ConfirmationAction.ACCEPT


# ---------------------------------------------------------------------------
# Level 1 - readback
# ---------------------------------------------------------------------------

class TestReadbackLevel:

    def test_moderate_confidence_reads_the_value_back(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.70, value="stomach pain")

        assert d.action is ConfirmationAction.CONFIRM
        assert d.level is ConfirmationLevel.READBACK
        assert "stomach pain" in d.utterance
        assert "Is that correct?" in d.utterance
        assert not d.is_terminal

    def test_readback_threshold_is_inclusive(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=CONFIRMATION_READBACK_THRESHOLD,
                                value="stomach pain")
        assert d.level is ConfirmationLevel.READBACK

    def test_just_below_readback_threshold_offers_choice(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=CONFIRMATION_READBACK_THRESHOLD - 0.01,
            value="stomach pain",
            candidates=TWO_CANDIDATES,
        )
        assert d.level is ConfirmationLevel.CHOICE

    def test_readback_falls_back_to_top_candidate_when_value_missing(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.70,
                                candidates=_candidates(("stomach pain", 0.70)))
        assert d.level is ConfirmationLevel.READBACK
        assert "stomach pain" in d.utterance


# ---------------------------------------------------------------------------
# Scenario B - level 2 choice confirmation
# ---------------------------------------------------------------------------

class TestChoiceLevel:

    def test_ambiguous_value_reads_candidates_for_patient_to_pick(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.50, value="stomach pain",
                                candidates=TWO_CANDIDATES)

        assert d.action is ConfirmationAction.CONFIRM
        assert d.level is ConfirmationLevel.CHOICE
        assert "stomach pain" in d.utterance
        assert "chest pain" in d.utterance
        assert ", or " in d.utterance
        assert [c["text"] for c in d.candidates_offered] == ["stomach pain", "chest pain"]

    def test_choice_threshold_is_inclusive(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=CONFIRMATION_CHOICE_THRESHOLD,
                                value="stomach pain", candidates=TWO_CANDIDATES)
        assert d.level is ConfirmationLevel.CHOICE

    def test_candidates_are_ordered_by_confidence(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.50,
            candidates=_candidates(("chest pain", 0.30), ("stomach pain", 0.55)),
        )
        assert [c["text"] for c in d.candidates_offered] == ["stomach pain", "chest pain"]

    def test_single_candidate_degrades_to_readback(self):
        # Cannot ask someone to choose between one option.
        d = decide_confirmation(PLAIN_FIELD, confidence=0.50, value="stomach pain",
                                candidates=_candidates(("stomach pain", 0.50)))
        assert d.level is ConfirmationLevel.READBACK

    def test_no_candidates_degrades_to_readback(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.50, value="stomach pain")
        assert d.level is ConfirmationLevel.READBACK

    def test_duplicate_candidate_texts_are_deduplicated(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.50,
            candidates=_candidates(("stomach pain", 0.50), ("Stomach pain", 0.45)),
        )
        # Only one distinct reading exists, so a choice question is impossible.
        assert d.level is ConfirmationLevel.READBACK
        assert d.utterance.lower().count("stomach pain") == 1

    def test_long_candidate_list_is_truncated_for_speech(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.50,
            candidates=_candidates(("stomach pain", 0.5), ("chest pain", 0.45),
                                   ("back pain", 0.4), ("hip pain", 0.35),
                                   ("head pain", 0.3)),
        )
        assert d.level is ConfirmationLevel.CHOICE
        assert len(d.candidates_offered) == MAX_CANDIDATES_TO_READ
        assert "head pain" not in d.utterance

    def test_three_candidates_render_readable_list(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.50,
            candidates=_candidates(("stomach pain", 0.5), ("chest pain", 0.45),
                                   ("back pain", 0.4)),
        )
        assert "stomach pain, chest pain, or back pain" in d.utterance


# ---------------------------------------------------------------------------
# Below the floor - first pass still tries to recover the value
# ---------------------------------------------------------------------------

class TestBelowFloorFirstPass:

    def test_very_low_confidence_first_pass_offers_choice(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.25, candidates=TWO_CANDIDATES)

        assert d.action is ConfirmationAction.CONFIRM
        assert d.level is ConfirmationLevel.CHOICE

    def test_very_low_confidence_without_candidates_tries_one_readback(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.25, value="stomach pain")
        assert d.level is ConfirmationLevel.READBACK

    def test_nothing_to_confirm_goes_straight_to_human(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.20)

        assert d.action is ConfirmationAction.MANUAL_REVIEW
        assert d.reason == "nothing_to_confirm"


# ---------------------------------------------------------------------------
# Scenario C - repeated failure terminates
# ---------------------------------------------------------------------------

class TestEscalationAndTermination:

    def test_critical_field_escalates_to_spelling_after_one_attempt(self):
        d = decide_confirmation(
            CRITICAL_FIELD,
            confidence=0.30,
            value="Metformin",
            candidates=TWO_CANDIDATES,
            attempted_levels=[ConfirmationLevel.CHOICE],
        )

        assert d.action is ConfirmationAction.CONFIRM
        assert d.level is ConfirmationLevel.SPELL
        assert d.reason == "escalated_to_spelling"
        assert d.is_critical
        assert "M as in Mike" in d.utterance

    def test_non_critical_field_goes_to_human_after_one_attempt(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.30,
            value="stomach pain",
            candidates=TWO_CANDIDATES,
            attempted_levels=[ConfirmationLevel.CHOICE],
        )

        assert d.action is ConfirmationAction.MANUAL_REVIEW
        assert d.reason == "low_confidence_after_confirmation"
        assert d.needs_hitl
        assert d.is_terminal
        assert not d.is_critical

    def test_manual_review_utterance_asks_nothing(self):
        d = decide_confirmation(PLAIN_FIELD, confidence=0.30, value="stomach pain",
                                attempted_levels=[ConfirmationLevel.READBACK])
        assert d.action is ConfirmationAction.MANUAL_REVIEW
        # Tells the patient it is handled; does not pose another question.
        assert "nurses" in d.utterance
        assert "?" not in d.utterance

    def test_critical_field_gives_up_after_spelling(self):
        d = decide_confirmation(
            CRITICAL_FIELD,
            confidence=0.30,
            value="Metformin",
            attempted_levels=[ConfirmationLevel.CHOICE, ConfirmationLevel.SPELL],
        )
        assert d.action is ConfirmationAction.MANUAL_REVIEW
        assert d.needs_hitl

    def test_non_critical_field_never_spells(self):
        # Exhausted both cheap levels at a confidence that is not "unheard".
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.70,
            value="stomach pain",
            attempted_levels=[ConfirmationLevel.READBACK, ConfirmationLevel.CHOICE],
        )
        assert d.action is ConfirmationAction.MANUAL_REVIEW
        assert d.level is not ConfirmationLevel.SPELL

    def test_attempt_budget_is_enforced_for_plain_fields(self):
        d = decide_confirmation(
            PLAIN_FIELD,
            confidence=0.70,
            value="stomach pain",
            attempted_levels=[ConfirmationLevel.READBACK] * MAX_CONFIRMATION_ATTEMPTS,
        )
        assert d.action is ConfirmationAction.MANUAL_REVIEW
        assert d.reason == "confirmation_attempts_exhausted"

    def test_critical_budget_is_one_larger(self):
        assert MAX_CRITICAL_CONFIRMATION_ATTEMPTS == MAX_CONFIRMATION_ATTEMPTS + 1


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------

class TestLadderInvariants:

    @pytest.mark.parametrize("field_name", [PLAIN_FIELD, CRITICAL_FIELD])
    @pytest.mark.parametrize("confidence", [0.0, 0.15, 0.30, 0.39, 0.45, 0.59,
                                            0.65, 0.80, 0.84])
    @pytest.mark.parametrize("with_candidates", [True, False])
    def test_every_trajectory_terminates(self, field_name, confidence, with_candidates):
        """
        Whatever the confidence, a patient is never asked forever.

        Drives the tracker with a permanently unhelpful confidence and
        asserts it reaches a terminal decision quickly.
        """
        tracker = ConfirmationTracker()
        candidates = TWO_CANDIDATES if with_candidates else None

        for _ in range(MAX_CRITICAL_CONFIRMATION_ATTEMPTS + 2):
            d = tracker.decide(field_name, confidence, value="stomach pain",
                               candidates=candidates)
            if d.is_terminal:
                assert d.action is ConfirmationAction.MANUAL_REVIEW
                break
        else:
            pytest.fail(f"ladder did not terminate for {field_name} @ {confidence}")

        # Bounded number of spoken confirmations.
        assert tracker.attempt_count(field_name) <= MAX_CRITICAL_CONFIRMATION_ATTEMPTS

    @pytest.mark.parametrize("field_name", [PLAIN_FIELD, CRITICAL_FIELD])
    def test_levels_never_repeat(self, field_name):
        tracker = ConfirmationTracker()
        for _ in range(6):
            d = tracker.decide(field_name, 0.30, value="Metformin",
                               candidates=TWO_CANDIDATES)
            if d.is_terminal:
                break
        history = tracker.history(field_name)
        assert len(history) == len(set(history)), f"level repeated: {history}"

    @pytest.mark.parametrize("field_name", [PLAIN_FIELD, CRITICAL_FIELD])
    def test_levels_only_escalate(self, field_name):
        tracker = ConfirmationTracker()
        for _ in range(6):
            d = tracker.decide(field_name, 0.50, value="Metformin",
                               candidates=TWO_CANDIDATES)
            if d.is_terminal:
                break
        history = tracker.history(field_name)
        assert history == sorted(history), f"ladder went backwards: {history}"


# ---------------------------------------------------------------------------
# Full escalation walks
# ---------------------------------------------------------------------------

class TestEscalationWalks:

    def test_critical_walk_choice_then_spell_then_human(self):
        tracker = ConfirmationTracker()

        first = tracker.decide(CRITICAL_FIELD, 0.30, value="Metformin",
                               candidates=TWO_CANDIDATES)
        assert first.level is ConfirmationLevel.CHOICE

        second = tracker.decide(CRITICAL_FIELD, 0.30, value="Metformin",
                                candidates=TWO_CANDIDATES)
        assert second.level is ConfirmationLevel.SPELL

        third = tracker.decide(CRITICAL_FIELD, 0.30, value="Metformin",
                               candidates=TWO_CANDIDATES)
        assert third.action is ConfirmationAction.MANUAL_REVIEW

        assert tracker.pending_manual_review() == [CRITICAL_FIELD]

    def test_plain_walk_choice_then_human(self):
        tracker = ConfirmationTracker()

        first = tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                               candidates=TWO_CANDIDATES)
        assert first.level is ConfirmationLevel.CHOICE

        second = tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                                candidates=TWO_CANDIDATES)
        assert second.action is ConfirmationAction.MANUAL_REVIEW
        assert tracker.attempt_count(PLAIN_FIELD) == 1

    def test_readback_then_resolved(self):
        tracker = ConfirmationTracker()

        first = tracker.decide(PLAIN_FIELD, 0.70, value="stomach pain")
        assert first.level is ConfirmationLevel.READBACK

        # Patient confirmed, re-score came back high.
        second = tracker.decide(PLAIN_FIELD, 0.95, value="stomach pain")
        assert second.action is ConfirmationAction.ACCEPT
        assert tracker.pending_manual_review() == []


# ---------------------------------------------------------------------------
# Tracker bookkeeping
# ---------------------------------------------------------------------------

class TestTracker:

    def test_accept_does_not_consume_budget(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.95, value="stomach pain")
        assert tracker.attempt_count(PLAIN_FIELD) == 0
        assert tracker.history(PLAIN_FIELD) == []

    def test_fields_are_tracked_independently(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.70, value="stomach pain")
        assert tracker.attempt_count("pain_level") == 0

    def test_manual_review_is_sticky(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                       candidates=TWO_CANDIDATES)
        second = tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                                candidates=TWO_CANDIDATES)
        assert second.action is ConfirmationAction.MANUAL_REVIEW

        # Asking again does not restart the ladder.
        third = tracker.decide(PLAIN_FIELD, 0.50, value="stomach pain",
                               candidates=TWO_CANDIDATES)
        assert third.action is ConfirmationAction.MANUAL_REVIEW
        assert tracker.attempt_count(PLAIN_FIELD) == 1

    def test_clear_late_answer_overrides_manual_review(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                       candidates=TWO_CANDIDATES)
        tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                       candidates=TWO_CANDIDATES)
        assert tracker.pending_manual_review() == [PLAIN_FIELD]

        revived = tracker.decide(PLAIN_FIELD, 0.95, value="stomach pain")
        assert revived.action is ConfirmationAction.ACCEPT
        assert tracker.pending_manual_review() == []

    def test_manual_review_decisions_carry_reasons(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                       candidates=TWO_CANDIDATES)
        tracker.decide(PLAIN_FIELD, 0.30, value="stomach pain",
                       candidates=TWO_CANDIDATES)

        decisions = tracker.manual_review_decisions()
        assert len(decisions) == 1
        assert decisions[0].reason == "low_confidence_after_confirmation"
        assert decisions[0].note

    def test_reset_single_field_and_all(self):
        tracker = ConfirmationTracker()
        tracker.decide(PLAIN_FIELD, 0.70, value="stomach pain")
        tracker.decide(CRITICAL_FIELD, 0.70, value="Metformin")

        tracker.reset(PLAIN_FIELD)
        assert tracker.attempt_count(PLAIN_FIELD) == 0
        assert tracker.attempt_count(CRITICAL_FIELD) == 1

        tracker.reset()
        assert tracker.attempt_count(CRITICAL_FIELD) == 0


# ---------------------------------------------------------------------------
# Field classification and spelling helpers
# ---------------------------------------------------------------------------

class TestFieldClassification:

    @pytest.mark.parametrize("name", [
        "patient_name", "medication_name", "medication_adherence",
        "medication_dosage", "allergy", "emergency_concerns",
    ])
    def test_enumerated_critical_fields(self, name):
        assert is_critical_field(name)

    @pytest.mark.parametrize("name", [
        "caregiver_name", "new_medication", "drug_switch", "daily_dose",
        "allergies_reported",
    ])
    def test_substring_fallback_fails_safe(self, name):
        assert is_critical_field(name)

    @pytest.mark.parametrize("name", [
        "symptom_update", "pain_level", "appointment_compliance", "", "mood",
    ])
    def test_non_critical_fields(self, name):
        assert not is_critical_field(name)

    def test_case_insensitive(self):
        assert is_critical_field("Patient_Name")

    def test_field_label_falls_back_to_readable_name(self):
        assert field_label("patient_name") == "your name"
        assert field_label("unknown_field") == "unknown field"


class TestSpelling:

    def test_letters_use_the_phonetic_alphabet(self):
        # Bare letters are the pairs a phone line destroys (B/D/P/T/V),
        # so the spelling tier reads "M as in Mike" instead.
        assert spell_out("Amlo") == (
            "A as in Alpha, M as in Mike, L as in Lima, O as in Oscar"
        )

    def test_plain_mode_is_available(self):
        assert spell_out("Amlo", phonetic=False) == "A, M, L, O"

    def test_digits_are_read_as_digits(self):
        assert spell_out("B12", phonetic=False) == "B, 1, 2"

    def test_punctuation_and_spaces_are_dropped(self):
        assert spell_out("A-B C", phonetic=False) == "A, B, C"

    def test_long_values_are_truncated(self):
        spelled = spell_out("Acetylsalicylicacid", phonetic=False)
        assert len(spelled.split(", ")) == MAX_LETTERS_TO_SPELL

    def test_spelling_utterance_names_the_field(self):
        d = decide_confirmation(
            "medication_name",
            confidence=0.30,
            value="Metformin",
            attempted_levels=[ConfirmationLevel.READBACK],
        )
        assert d.level is ConfirmationLevel.SPELL
        assert "the name of the medication" in d.utterance
        assert "M as in Mike" in d.utterance
        assert "E as in Echo" in d.utterance

    def test_unspellable_value_falls_back_to_readback_text(self):
        d = decide_confirmation(
            "patient_name",
            confidence=0.30,
            value="???",
            attempted_levels=[ConfirmationLevel.READBACK],
        )
        # Still a SPELL-level decision, but the wording degrades gracefully
        # rather than reading out an empty letter list.
        assert d.level is ConfirmationLevel.SPELL
        assert "as in" not in d.utterance
