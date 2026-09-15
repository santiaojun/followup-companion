"""
Tiered confirmation - deciding *how hard* to re-confirm a value we may have
misheard.

Scope: this module handles one kind of ambiguity only - uncertainty about
what words the patient actually said. The other kind (heard perfectly, too
vague to use) is not a confirmation problem at all and is handled
separately in agents/symptom_probe.py.

The problem
-----------
A telephone follow-up agent constantly half-hears things. The naive fix is to
re-ask whenever confidence is low, which burns the patient's patience on
values that were fine, and the expensive fix (spell everything out) is worse.

So confirmation is graded by confidence, and each level costs the patient
more time than the one below it:

    confidence      level               what the patient hears
    -------------------------------------------------------------------------
    >= 0.85         NONE                nothing - the value is taken as heard
    0.60 - 0.85     READBACK            "You said X, is that correct?"
    0.40 - 0.60     CHOICE              "Did you say stomach pain, or chest
                                         pain?"
    < 0.40          (see below)

Below 0.40 the value is not really "heard" at all, so one attempt is spent
trying to recover it (CHOICE when the extractor gave us candidates, otherwise
READBACK). If confidence is *still* below 0.40 after that attempt:

    critical field      -> SPELL, letter by letter (name, drug name, dose)
    non-critical field  -> stop asking, mark for human review

That last line is the point. Re-asking a patient a third and fourth time
does not produce better data, it just produces a worse call, so the ladder
gives up on purpose and hands the field to triage/ instead.

Termination guarantees
----------------------
Two independent mechanisms, either of which alone would be enough:

  1. Monotonic escalation. A field never revisits a level it already used
     (NONE -> READBACK -> CHOICE -> SPELL -> manual review), so at most
     three utterances exist in the whole ladder.
  2. Attempt budget. MAX_CONFIRMATION_ATTEMPTS (2) utterances per field,
     plus at most one extra SPELL pass for critical fields only.

Usage
-----
    tracker = ConfirmationTracker()

    d = tracker.decide("medication_name", confidence=0.52,
                       candidates=[{"text": "Metformin", "confidence": 0.52},
                                   {"text": "Metronidazole", "confidence": 0.48}])
    if d.action is ConfirmationAction.CONFIRM:
        say(d.utterance)                  # patient picks one
        d2 = tracker.decide("medication_name", confidence=0.35, ...)
        # -> SPELL, because medication_name is critical

    for field_name in tracker.pending_manual_review():
        ...                               # hand to triage/

`decide_confirmation()` is the pure function underneath; ConfirmationTracker
only adds the per-field attempt history. Test the pure function when you can.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from agents.models import (
    CONFIRMATION_ACCEPT_THRESHOLD,
    CONFIRMATION_CHOICE_THRESHOLD,
    CONFIRMATION_READBACK_THRESHOLD,
    CRITICAL_FIELD_KEYWORDS,
    CRITICAL_FIELDS,
    FIELD_LABELS,
    MAX_CONFIRMATION_ATTEMPTS,
    MIN_CANDIDATES_FOR_CHOICE,
    ConfirmationAction,
    ConfirmationDecision,
    ConfirmationLevel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Critical fields may spend one extra attempt on top of the normal budget,
# and only on SPELL. Since SPELL can be used at most once (escalation is
# monotonic), this cannot loop - worst case is three utterances per field.
MAX_CRITICAL_CONFIRMATION_ATTEMPTS: int = MAX_CONFIRMATION_ATTEMPTS + 1

# A spoken choice list stops being usable past three options. Extra
# candidates are dropped (lowest confidence first) rather than read out.
MAX_CANDIDATES_TO_READ: int = 3

# NATO phonetic alphabet, used for the spelling tier.
#
# Bare letters are the wrong tool on a phone line: B/D/P/T/V/E and M/N/F/S
# are the exact pairs a narrowband codec destroys, which is how you end up
# confirming the wrong drug with total confidence on both sides. Reading
# "M as in Mike" removes the confusion.
PHONETIC_ALPHABET: dict[str, str] = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo",
    "F": "Foxtrot", "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliet",
    "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November", "O": "Oscar",
    "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango",
    "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray",
    "Y": "Yankee", "Z": "Zulu",
}

# Spelling out a long value on the phone is worse than not doing it; past
# this length we confirm the opening letters and stop.
MAX_LETTERS_TO_SPELL: int = 12


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

def is_critical_field(field_name: str) -> bool:
    """
    True when getting this field wrong is consequential enough to justify
    letter-by-letter confirmation.

    Exact membership in CRITICAL_FIELDS first, then a substring check against
    CRITICAL_FIELD_KEYWORDS so field names we have not enumerated yet
    ("new_medication_name", "caregiver_name") fail safe as critical rather
    than being silently demoted to manual review.
    """
    name = (field_name or "").strip().lower()
    if name in CRITICAL_FIELDS:
        return True
    return any(keyword in name for keyword in CRITICAL_FIELD_KEYWORDS)


def field_label(field_name: str) -> str:
    """How to refer to a field out loud; falls back to the raw name."""
    return FIELD_LABELS.get(field_name, field_name.replace("_", " "))


# ---------------------------------------------------------------------------
# Utterance construction
# ---------------------------------------------------------------------------

def spell_out(value: str, phonetic: bool = True) -> str:
    """
    Render a value for letter-by-letter verification.

    Letters become "M as in Mike" by default; digits are read as digits and
    anything else (hyphens, spaces) is dropped since it carries no
    information when read aloud. Truncated at MAX_LETTERS_TO_SPELL.
    """
    parts: list[str] = []
    for ch in str(value):
        upper = ch.upper()
        if upper in PHONETIC_ALPHABET:
            parts.append(f"{upper} as in {PHONETIC_ALPHABET[upper]}" if phonetic else upper)
        elif ch.isdigit():
            parts.append(ch)
        if len(parts) >= MAX_LETTERS_TO_SPELL:
            break
    return ", ".join(parts)


def _readback_utterance(display: str) -> str:
    return f"Let me make sure I have that right - you said {display}. Is that correct?"


def _choice_utterance(texts: Sequence[str]) -> str:
    if len(texts) == 2:
        options = f"{texts[0]}, or {texts[1]}"
    else:
        options = ", ".join(texts[:-1]) + f", or {texts[-1]}"
    return f"Sorry, I didn't quite catch that. Did you say {options}?"


def _spell_utterance(field_name: str, display: str) -> str:
    label = field_label(field_name)
    spelled = spell_out(display)
    if not spelled:
        # Nothing spellable (e.g. punctuation only); fall back to a read-back.
        return _readback_utterance(display)
    return (
        f"I want to be sure I have {label} exactly right, so let me spell it "
        f"back to you: {spelled}. Have I got that right?"
    )


def _manual_review_utterance(field_name: str) -> str:
    """
    Closing line for a field we are giving up on.

    Deliberately does not ask anything: the patient is told it will be
    handled, which is the whole reason we stop instead of looping.
    """
    label = field_label(field_name)
    return (
        f"No problem - I'll make a note about {label} and have one of our "
        f"nurses confirm it with you later. I won't keep you on the line for it."
    )


# ---------------------------------------------------------------------------
# Candidate handling
# ---------------------------------------------------------------------------

def _rank_candidates(candidates: Optional[Iterable[dict]]) -> list[dict]:
    """
    Sort candidates by descending confidence and drop duplicate texts.

    De-duplication matters: two candidates with the same text make a CHOICE
    question nonsensical ("Did you say chest pain, or chest pain?"), so
    identical texts collapse to one and the field falls back to a read-back.
    """
    if not candidates:
        return []
    ranked = sorted(
        (c for c in candidates if str(c.get("text", "")).strip()),
        key=lambda c: float(c.get("confidence", 0.0)),
        reverse=True,
    )
    seen: set[str] = set()
    unique: list[dict] = []
    for c in ranked:
        text = str(c["text"]).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"text": text, "confidence": float(c.get("confidence", 0.0))})
    return unique


def _display_value(value: Any, ranked: list[dict]) -> str:
    """
    The string to speak back to the patient.

    Prefers the explicit extracted value; falls back to the top candidate so
    a read-back is still possible when only candidates were supplied.
    """
    if value is not None and str(value).strip():
        return str(value).strip()
    if ranked:
        return ranked[0]["text"]
    return ""


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def decide_confirmation(
    field_name: str,
    confidence: float,
    value: Any = None,
    candidates: Optional[Iterable[dict]] = None,
    attempted_levels: Sequence[ConfirmationLevel] = (),
) -> ConfirmationDecision:
    """
    Decide what to do about one value, given what has already been tried.

    Pure function: no state, no I/O, no LLM. All history arrives through
    `attempted_levels` (the levels already spoken for this field, in order),
    which is what ConfirmationTracker maintains.

    Parameters
    ----------
    field_name       Used for criticality and for the spoken label.
    confidence       Current confidence in [0.0, 1.0]. After a confirmation
                     attempt, pass the *re-scored* confidence, not the
                     original one.
    value            Best single value, if any. Used for READBACK / SPELL.
    candidates       Extractor candidates, [{"text", "confidence"}, ...].
                     Two or more distinct ones enable CHOICE.
    attempted_levels Levels already used for this field.

    Returns
    -------
    ConfirmationDecision with action ACCEPT (use it), CONFIRM (speak
    `utterance` and re-score) or MANUAL_REVIEW (stop, route to triage/).
    """
    critical = is_critical_field(field_name)
    ranked = _rank_candidates(candidates)
    display = _display_value(value, ranked)
    attempted = tuple(attempted_levels)
    attempts = len(attempted)
    highest = max(attempted) if attempted else ConfirmationLevel.NONE

    def manual(reason: str, note: str) -> ConfirmationDecision:
        return ConfirmationDecision(
            field_name=field_name,
            action=ConfirmationAction.MANUAL_REVIEW,
            level=highest,
            utterance=_manual_review_utterance(field_name),
            reason=reason,
            note=note,
            candidates_offered=ranked,
            attempt_index=attempts,
            is_critical=critical,
        )

    def confirm(level: ConfirmationLevel, utterance: str, reason: str, note: str,
                offered: Optional[list[dict]] = None) -> ConfirmationDecision:
        return ConfirmationDecision(
            field_name=field_name,
            action=ConfirmationAction.CONFIRM,
            level=level,
            utterance=utterance,
            reason=reason,
            note=note,
            candidates_offered=offered if offered is not None else ranked,
            attempt_index=attempts,
            is_critical=critical,
        )

    def escalate(reason: str, note: str) -> ConfirmationDecision:
        """
        Last resort for a field the cheaper levels could not settle.

        Critical fields get one spelling pass (if they have not had one and
        there is something to spell); everything else goes to a human.
        """
        if critical and ConfirmationLevel.SPELL not in attempted and display:
            return confirm(
                ConfirmationLevel.SPELL,
                _spell_utterance(field_name, display),
                reason="escalated_to_spelling",
                note=(
                    f"Critical field still unresolved ({note}); verifying "
                    f"letter by letter."
                ),
            )
        return manual(reason, note)

    # -- 1. Trust band -----------------------------------------------------
    # High confidence needs no utterance, even mid-ladder: if the patient
    # volunteered the value clearly on a retry, that settles it.
    if confidence >= CONFIRMATION_ACCEPT_THRESHOLD:
        return ConfirmationDecision(
            field_name=field_name,
            action=ConfirmationAction.ACCEPT,
            level=ConfirmationLevel.NONE,
            utterance="",
            reason="confidence_above_accept_threshold",
            note=(
                f"Confidence {confidence:.2f} >= "
                f"{CONFIRMATION_ACCEPT_THRESHOLD:.2f}; accepted without confirmation."
            ),
            candidates_offered=ranked,
            attempt_index=attempts,
            is_critical=critical,
        )

    # -- 2. Attempt budget -------------------------------------------------
    budget = (
        MAX_CRITICAL_CONFIRMATION_ATTEMPTS if critical
        else MAX_CONFIRMATION_ATTEMPTS
    )
    if attempts >= budget:
        return manual(
            reason="confirmation_attempts_exhausted",
            note=(
                f"{attempts} confirmation attempt(s) already spent "
                f"(budget {budget}), confidence still {confidence:.2f}. "
                f"Not asking again."
            ),
        )

    # -- 3. Already tried once and still below the "not heard" floor -------
    # This is the rule that stops the loop early: another read-back of a
    # value we cannot hear will not work, so escalate or hand over now.
    if confidence < CONFIRMATION_CHOICE_THRESHOLD and attempts >= 1:
        return escalate(
            reason="low_confidence_after_confirmation",
            note=(
                f"Confidence {confidence:.2f} still below "
                f"{CONFIRMATION_CHOICE_THRESHOLD:.2f} after {attempts} attempt(s)."
            ),
        )

    # -- 4. Ladder exhausted ------------------------------------------------
    if highest >= ConfirmationLevel.SPELL:
        return manual(
            reason="confirmation_ladder_exhausted",
            note="Spelling confirmation already attempted and still unresolved.",
        )

    # -- 5. Baseline level from the confidence band ------------------------
    if confidence >= CONFIRMATION_READBACK_THRESHOLD:
        want = ConfirmationLevel.READBACK
    else:
        # Covers both 0.40-0.60 and the first pass below 0.40: a spoken
        # choice is the cheapest thing that can actually recover the value.
        want = ConfirmationLevel.CHOICE

    # -- 6. Monotonic escalation - never repeat a level --------------------
    if want <= highest:
        want = ConfirmationLevel(int(highest) + 1)

    # -- 7. Nothing to work with at all ------------------------------------
    # An empty display means no value *and* no candidates (the display falls
    # back to the top candidate), so there is no utterance to build at any
    # level. Hand it over rather than inventing something to read back.
    if not display:
        return escalate(
            reason="nothing_to_confirm",
            note="No extracted value and no candidates to read back.",
        )

    # -- 8. CHOICE requires something to choose between --------------------
    if want is ConfirmationLevel.CHOICE and len(ranked) < MIN_CANDIDATES_FOR_CHOICE:
        if ConfirmationLevel.READBACK > highest:
            # Only one reading exists; confirm that one instead.
            want = ConfirmationLevel.READBACK
        else:
            return escalate(
                reason="no_candidates_to_offer",
                note=(
                    f"Confidence {confidence:.2f} needs a choice question but "
                    f"only {len(ranked)} distinct candidate(s) available."
                ),
            )

    # -- 9. Render ----------------------------------------------------------
    if want is ConfirmationLevel.READBACK:
        return confirm(
            ConfirmationLevel.READBACK,
            _readback_utterance(display),
            reason="readback_confirmation",
            note=(
                f"Confidence {confidence:.2f} in "
                f"[{CONFIRMATION_READBACK_THRESHOLD:.2f}, "
                f"{CONFIRMATION_ACCEPT_THRESHOLD:.2f}); single read-back."
            ),
        )

    if want is ConfirmationLevel.CHOICE:
        offered = ranked[:MAX_CANDIDATES_TO_READ]
        return confirm(
            ConfirmationLevel.CHOICE,
            _choice_utterance([c["text"] for c in offered]),
            reason="candidate_choice_confirmation",
            note=(
                f"Confidence {confidence:.2f} below "
                f"{CONFIRMATION_READBACK_THRESHOLD:.2f}; reading "
                f"{len(offered)} candidate(s) for the patient to pick."
            ),
            offered=offered,
        )

    # want is SPELL
    if not critical:
        # Spelling is reserved for fields where the cost is justified.
        return manual(
            reason="no_cheaper_confirmation_available",
            note=(
                f"Non-critical field exhausted cheaper levels "
                f"(tried {[lvl.name for lvl in attempted]}); "
                f"routing to human review."
            ),
        )
    return confirm(
        ConfirmationLevel.SPELL,
        _spell_utterance(field_name, display),
        reason="escalated_to_spelling",
        note=f"Critical field unresolved at confidence {confidence:.2f}; spelling out.",
    )


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------

class ConfirmationTracker:
    """
    Per-call confirmation state: which levels each field has already used.

    One instance per call. Stateless across calls by design - confirmation
    history is turn-level control flow, not something to persist into the
    patient's profile.
    """

    def __init__(self) -> None:
        self._attempts: dict[str, list[ConfirmationLevel]] = {}
        self._manual_review: dict[str, ConfirmationDecision] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide(
        self,
        field_name: str,
        confidence: float,
        value: Any = None,
        candidates: Optional[Iterable[dict]] = None,
    ) -> ConfirmationDecision:
        """
        Decide the next confirmation step for a field and record the attempt.

        Calling this repeatedly with a still-low confidence walks the ladder
        and eventually returns MANUAL_REVIEW; it never returns CONFIRM
        forever.
        """
        # A field already handed to a human stays handed over, unless the
        # patient has since said it clearly enough to stand on its own.
        if field_name in self._manual_review:
            if confidence >= CONFIRMATION_ACCEPT_THRESHOLD:
                del self._manual_review[field_name]
            else:
                return self._manual_review[field_name]

        decision = decide_confirmation(
            field_name=field_name,
            confidence=confidence,
            value=value,
            candidates=candidates,
            attempted_levels=self._attempts.get(field_name, ()),
        )

        # Only spoken confirmation attempts count against the budget.
        if decision.action is ConfirmationAction.CONFIRM:
            self._attempts.setdefault(field_name, []).append(decision.level)
        elif decision.action is ConfirmationAction.MANUAL_REVIEW:
            self._manual_review[field_name] = decision

        return decision

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def history(self, field_name: str) -> list[ConfirmationLevel]:
        """Levels already spoken for this field, in order."""
        return list(self._attempts.get(field_name, []))

    def attempt_count(self, field_name: str) -> int:
        return len(self._attempts.get(field_name, ()))

    def pending_manual_review(self) -> list[str]:
        """Fields the ladder gave up on; hand these to triage/."""
        return sorted(self._manual_review)

    def manual_review_decisions(self) -> list[ConfirmationDecision]:
        """Full decisions for the given-up fields, for the audit trail."""
        return [self._manual_review[f] for f in sorted(self._manual_review)]

    def reset(self, field_name: Optional[str] = None) -> None:
        """Clear history for one field, or for all fields when name is None."""
        if field_name is None:
            self._attempts.clear()
            self._manual_review.clear()
            return
        self._attempts.pop(field_name, None)
        self._manual_review.pop(field_name, None)
