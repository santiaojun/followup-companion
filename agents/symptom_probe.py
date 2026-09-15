"""
Symptom detail probing - the *other* kind of ambiguity.

Two things get called "unclear" in a follow-up call, and they need opposite
treatment. These are two independent mechanisms, not two settings of one.

  1. We are unsure what words were said.
     The patient may have said "stomach pain" or "chest pain" - a
     transcription problem. Fixed by agents/confirmation.py: read it back,
     offer a choice, spell it out.

  2. We are certain what was said, and it still is not usable.
     "My arm hurts" is transcribed perfectly at confidence 1.0. There is no
     ambiguity about the words at all. It is simply missing the site, the
     character and the duration - the things that decide whether this is
     nothing or a referred cardiac pain.

Reading "my arm hurts" back to the patient (case 1's tool) does nothing for
case 2. The fix for case 2 is a clinical follow-up question, chosen from a
checklist for that symptom family.

What triggers a probe
---------------------
Primarily `ExtractedClaim.needs_clarification`, set by the extractor during
extraction - an LLM reading the whole transcript judges "symptom described
but incomplete" far better than any keyword list can.

When that flag is absent or False, is_symptom_incomplete() falls back to
checking the text itself against the checklist below. The fallback keeps
this module useful with extractors that do not set the flag yet, but the
flag is the authoritative signal.

classify_ambiguity() is the fork between the two mechanisms, and it
resolves case 1 first: you cannot ask "where exactly does it hurt" about a
sentence you did not hear.

The registry
------------
PROBE_REGISTRY maps a SymptomCategory to its ordered dimension checklist.
Four families are covered so far - pain, respiratory, gastrointestinal and
wound - plus a generic fallback for anything unrecognised. Adding a family
is one entry in the two tables below (_CATEGORY_KEYWORDS and
PROBE_REGISTRY); no logic changes.

Only the `required` dimensions decide whether a description counts as
incomplete; they are the minimum that makes a report clinically actionable
(for pain: site, character, duration). The rest are worth asking when there
is room, but their absence does not make the answer unusable.

Each dimension knows how to detect that the patient already covered it, so
a patient who volunteers "sharp pain in my left outer forearm, on and off
since Tuesday" is not asked where, when, or what kind. We only spend
questions on the gaps.

Matching
--------
Keywords match case-insensitively at a word boundary, as prefixes: "breath"
covers "breathless" and "breathing", "constipat" covers "constipation" and
"constipated", and "sore" does not fire on "score".

Usage
-----
    plan = build_probe_plan("symptom_update", "my arm hurts")
    plan.category      # SymptomCategory.PAIN
    plan.missing       # ['location', 'quality', 'duration', ...]
    plan.questions     # first MAX_PROBE_QUESTIONS of them, as spoken text

    # Driven off extractor output:
    if classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED:
        for q in plan_for_claim(claim).questions:
            say(q)
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

from agents.models import (
    CONFIRMATION_ACCEPT_THRESHOLD,
    MAX_PROBE_QUESTIONS,
    SYMPTOM_FIELDS,
    AmbiguityKind,
    ProbeDimension,
    SymptomCategory,
    SymptomProbePlan,
)

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _any_pattern(keywords: tuple[str, ...]) -> Optional[re.Pattern]:
    """One alternation matching any keyword as a word-initial prefix."""
    if not keywords:
        return None
    alternation = "|".join(re.escape(k) for k in keywords)
    return re.compile(rf"\b(?:{alternation})", re.IGNORECASE)


@lru_cache(maxsize=None)
def _each_pattern(keywords: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """One pattern per keyword, for counting how many distinct ones hit."""
    return tuple(re.compile(rf"\b{re.escape(k)}", re.IGNORECASE) for k in keywords)


# ---------------------------------------------------------------------------
# Shared detection patterns
# ---------------------------------------------------------------------------

# "3 days", "2 weeks", "48 hours" - a quantified time span.
_DURATION_PATTERN = r"\d+\s*(?:hours?|days?|weeks?|months?|years?|minutes?)"

# "6 out of 10", "7/10", "a 6" - a quantified severity.
_SEVERITY_PATTERN = (
    r"\b(?:10|[0-9])\s*/\s*10\b"
    r"|\b(?:10|[0-9])\s*out of\s*10\b"
    r"|\b(?:10|[0-9])\s*(?:on|out of)\s*(?:the\s*)?(?:scale|ten)\b"
)

# "3 times a day", "twice", "once or twice" - a quantified frequency.
_FREQUENCY_PATTERN = (
    r"\d+\s*times?"
    r"|\b(?:once|twice|three times|four times|five times)\b"
)

# Answers that are complete as they stand and must NOT be probed. Without
# this, "no new symptoms" would earn the patient three follow-up questions
# about a symptom they just said they do not have.
_DENIAL_PATTERN = re.compile(
    r"\b(?:none|nothing|no new|nothing new|nothing else|no change|no changes"
    r"|unchanged|no different|about the same|same as before|same as last time"
    r"|no problems|no issues|no complaints|no symptoms|no concerns"
    r"|denies|denied|stable|normal|not really|all good|all fine"
    r"|don't have any|do not have any|haven't had any|nothing at all)\b",
    re.IGNORECASE,
)

# Positive self-reports, which only count as a denial when nothing negates
# them - "doing well" ends the topic, "not doing well" does not.
_POSITIVE_PATTERN = re.compile(
    r"\b(?:i'm fine|im fine|feeling fine|feeling good|feeling well"
    r"|doing well|doing fine|doing good|doing ok|doing okay"
    r"|pretty good|quite good|much better|everything's fine"
    r"|everything is fine)\b",
    re.IGNORECASE,
)

_NEGATOR_PATTERN = re.compile(
    r"\bnot\b|n't\b|\bnever\b|\bhardly\b|\bbarely\b|\bno longer\b",
    re.IGNORECASE,
)

# A bare scale answer ("6", "6/10", "a 7") - already the precise value the
# protocol asked for, so there is nothing vague about it.
_BARE_SCALE_PATTERN = re.compile(
    r"^\s*(?:a\s+)?(?:10|[0-9])(?:\.\d+)?\s*"
    r"(?:/\s*10|out of\s*10|out of ten)?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------
# Keyword hit count picks the category; ties go to _CATEGORY_PRECEDENCE.
# Keywords inside a family avoid being prefixes of each other so the count
# stays meaningful ("hurt" already covers "hurts" and "hurting").

_CATEGORY_KEYWORDS: dict[SymptomCategory, tuple[str, ...]] = {
    SymptomCategory.PAIN: (
        "pain", "ache", "aching", "hurt", "sore", "tender", "cramp",
        "colic", "stabbing", "throbbing", "stiff",
        # Patients routinely downgrade pain to "discomfort", especially
        # cardiac pain. Treating it as pain is the safe reading.
        "discomfort", "uncomfortable", "twinge",
    ),
    SymptomCategory.RESPIRATORY: (
        "cough", "wheez", "breath", "dyspnea", "dyspnoea", "sputum",
        "phlegm", "mucus", "winded", "chest tight", "tight chest",
        "suffocat",
    ),
    SymptomCategory.GASTROINTESTINAL: (
        "nausea", "nauseous", "vomit", "throw up", "threw up", "throwing up",
        "diarrhea", "diarrhoea", "constipat", "stool", "bowel", "appetite",
        "bloat", "heartburn", "stomach", "indigestion", "gassy", "reflux",
        "belch", "loose motions",
    ),
    SymptomCategory.WOUND: (
        "wound", "incision", "suture", "stitch", "staple", "scar",
        "surgical site", "dressing", "pus", "healing",
    ),
}

# Applied on a tie. Wound and respiratory outrank pain because their
# checklists carry the time-critical signs (dehiscence, infection,
# orthopnoea); "my incision hurts" is better served by the wound checklist
# than by a generic pain one. Pain outranks GI so that "stomach pain" is
# characterised as pain (site/character/duration) rather than as a bowel
# complaint.
_CATEGORY_PRECEDENCE: tuple[SymptomCategory, ...] = (
    SymptomCategory.WOUND,
    SymptomCategory.RESPIRATORY,
    SymptomCategory.PAIN,
    SymptomCategory.GASTROINTESTINAL,
)


def classify_symptom(text: str) -> SymptomCategory:
    """
    Pick the symptom family whose checklist fits this description.

    Returns SymptomCategory.UNKNOWN when nothing matches, which still gets a
    generic checklist rather than no follow-up at all.
    """
    raw = (text or "").strip()
    if not raw:
        return SymptomCategory.UNKNOWN

    scores = {
        category: sum(1 for p in _each_pattern(keywords) if p.search(raw))
        for category, keywords in _CATEGORY_KEYWORDS.items()
    }
    best = max(scores.values())
    if best == 0:
        return SymptomCategory.UNKNOWN

    for category in _CATEGORY_PRECEDENCE:
        if scores.get(category, 0) == best:
            return category
    return SymptomCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Shared dimension keyword sets
# ---------------------------------------------------------------------------

# A body region on its own ("my arm") is not a site - it does not say which
# arm or which part of it. Location counts as covered only when the patient
# also gave a qualifier, which is why "my arm hurts" still gets asked.
_LOCATION_QUALIFIERS: tuple[str, ...] = (
    "left", "right", "both", "upper", "lower", "inner", "inside",
    "outer", "outside", "front", "back of", "middle", "side of",
    "medial", "lateral", "proximal", "distal", "above", "below",
    "behind", "around my", "top of", "base of", "joint",
)

# Deliberately excludes vague time references ("recently", "lately", "for a
# while", "a bit now"). Those are exactly the answers that still need a
# duration question, so counting them as covered would defeat the purpose
# of asking.
_DURATION_KEYWORDS: tuple[str, ...] = (
    "day", "week", "month", "year", "hour", "minute",
    "yesterday", "last night", "this morning", "overnight",
    "constant", "intermittent", "sudden", "started", "began",
    "on and off", "comes and goes", "all the time", "ever since",
)

_ASSOCIATED_KEYWORDS: tuple[str, ...] = (
    "also", "along with", "as well", "besides that", "at the same time",
    "fever", "temperature", "nausea", "dizz", "numb", "tingl", "swell",
    "sweat", "palpitat", "chills", "vomit",
)

PROBE_REGISTRY: dict[SymptomCategory, tuple[ProbeDimension, ...]] = {

    # ---- Pain ----------------------------------------------------------
    SymptomCategory.PAIN: (
        ProbeDimension(
            key="location",
            question=(
                "Whereabouts exactly is it - and is that on the left side "
                "or the right?"
            ),
            required=True,
            present_keywords=_LOCATION_QUALIFIERS,
        ),
        ProbeDimension(
            key="quality",
            question=(
                "What does it feel like - sharp, dull, burning, or more of "
                "an ache?"
            ),
            required=True,
            present_keywords=(
                "sharp", "dull", "burn", "stab", "throb", "shoot", "gnaw",
                "sting", "cramp", "press", "squeez", "tight", "tearing",
                "pins and needles", "aching",
            ),
        ),
        ProbeDimension(
            key="duration",
            question=(
                "How long have you had it, and is it there all the time or "
                "does it come and go?"
            ),
            required=True,
            present_keywords=_DURATION_KEYWORDS,
            present_pattern=_DURATION_PATTERN,
        ),
        ProbeDimension(
            key="associated",
            question=(
                "Have you noticed anything else along with it, like a fever, "
                "nausea, or any numbness?"
            ),
            required=False,
            present_keywords=_ASSOCIATED_KEYWORDS,
        ),
        ProbeDimension(
            key="severity",
            question=(
                "On a scale where 0 is no pain and 10 is the worst you can "
                "imagine, where is it right now?"
            ),
            required=False,
            present_keywords=("mild", "moderate", "severe", "unbearable",
                              "excruciating", "slight", "terrible", "awful",
                              "killing me", "bearable"),
            present_pattern=_SEVERITY_PATTERN,
        ),
        ProbeDimension(
            key="triggers",
            question=(
                "Is there anything that makes it worse or better - moving "
                "about, resting, or your medication?"
            ),
            required=False,
            present_keywords=("worse", "better", "rest", "walk", "mov",
                              "bend", "lying", "lie down", "press",
                              "exert", "stairs", "weather", "medication"),
        ),
    ),

    # ---- Respiratory ---------------------------------------------------
    SymptomCategory.RESPIRATORY: (
        ProbeDimension(
            key="trigger",
            question=(
                "What brings it on - walking, going up stairs, or when you "
                "lie down?"
            ),
            required=True,
            present_keywords=("walk", "stair", "climb", "lying", "lie down",
                              "lay down", "flat", "exert", "activity",
                              "at night", "rest", "uphill", "carrying",
                              "housework", "chores"),
        ),
        ProbeDimension(
            key="onset_duration",
            question=(
                "When did it start, and how many days has it been going on?"
            ),
            required=True,
            present_keywords=_DURATION_KEYWORDS,
            present_pattern=_DURATION_PATTERN,
        ),
        ProbeDimension(
            key="sputum",
            question=(
                "When you cough, are you bringing anything up? What colour "
                "is it, and is there any blood?"
            ),
            required=False,
            present_keywords=("sputum", "phlegm", "mucus", "dry cough",
                              "bringing up", "coughing up", "blood",
                              "yellow", "green", "clear", "rust"),
        ),
        ProbeDimension(
            key="functional_impact",
            question=(
                "How far can you walk before you have to stop? Can you "
                "finish a sentence without catching your breath?"
            ),
            required=False,
            present_keywords=("step", "flight", "sentence", "talking",
                              "sleep", "block", "metre", "meter", "yard",
                              "mile", "stop", "pillow"),
            present_pattern=_SEVERITY_PATTERN,
        ),
        ProbeDimension(
            key="associated",
            question=(
                "Along with that, any fever, chest pain, or swelling in "
                "your ankles?"
            ),
            required=False,
            present_keywords=_ASSOCIATED_KEYWORDS,
        ),
    ),

    # ---- Gastrointestinal ----------------------------------------------
    SymptomCategory.GASTROINTESTINAL: (
        ProbeDimension(
            key="frequency",
            question="How many times a day is that happening?",
            required=True,
            present_keywords=("daily", "every day", "a day", "an hour",
                              "occasional", "frequent", "constantly",
                              "every time", "most days"),
            present_pattern=_FREQUENCY_PATTERN,
        ),
        ProbeDimension(
            key="duration",
            question="How many days has this been going on?",
            required=True,
            present_keywords=_DURATION_KEYWORDS,
            present_pattern=_DURATION_PATTERN,
        ),
        ProbeDimension(
            key="stool_appearance",
            question=(
                "What are your stools like - loose or formed? Any black "
                "colour or blood?"
            ),
            required=True,
            present_keywords=("loose", "watery", "formed", "hard", "black",
                              "tarry", "blood", "colour", "color", "mucus",
                              "pellet", "float"),
        ),
        ProbeDimension(
            key="meal_relation",
            question=(
                "Does it have anything to do with eating - worse after "
                "meals, or on an empty stomach?"
            ),
            required=False,
            present_keywords=("after eating", "before eating", "after meal",
                              "before meal", "with food", "empty stomach",
                              "fasting", "breakfast", "lunch", "dinner",
                              "when i eat"),
        ),
        ProbeDimension(
            key="intake",
            question=(
                "Are you able to keep food down at the moment, and are you "
                "drinking normally?"
            ),
            required=False,
            present_keywords=("appetite", "keep down", "keeping down",
                              "drinking", "fluid", "weight", "eating well",
                              "lost weight"),
        ),
        ProbeDimension(
            key="dehydration",
            question=(
                "Have you felt dry-mouthed or weak, or noticed you are "
                "passing less urine?"
            ),
            required=False,
            present_keywords=("dry mouth", "thirsty", "weak", "urine",
                              "peeing", "dehydrat", "light-headed",
                              "lightheaded"),
        ),
    ),

    # ---- Wound / incision (post-op) ------------------------------------
    SymptomCategory.WOUND: (
        ProbeDimension(
            key="appearance",
            question=(
                "Does the skin around it look red or swollen at all?"
            ),
            required=True,
            present_keywords=("red", "swollen", "swelling", "bruis",
                              "purple", "pale", "colour", "color",
                              "inflamed", "puffy", "warm to"),
        ),
        ProbeDimension(
            key="discharge",
            question=(
                "Is anything draining from it - any fluid or pus? What "
                "colour is it?"
            ),
            required=True,
            present_keywords=("drain", "discharge", "pus", "ooz", "fluid",
                              "weep", "wet", "bleed", "blood", "seep",
                              "smell"),
        ),
        ProbeDimension(
            key="fever",
            question=(
                "Have you had a fever in the last day or two? Have you "
                "taken your temperature?"
            ),
            required=True,
            present_keywords=("fever", "temperature", "chills", "shiver",
                              "hot", "febrile", "degrees", "sweats"),
        ),
        ProbeDimension(
            key="pain_trend",
            question=(
                "Is the soreness settling down, or has it got worse over "
                "the last few days?"
            ),
            required=False,
            present_keywords=("worse", "better", "improv", "settling",
                              "same", "increasing", "decreasing", "less",
                              "more"),
        ),
        ProbeDimension(
            key="integrity",
            question=(
                "Has any part of it opened up, or have any stitches come "
                "loose?"
            ),
            required=False,
            present_keywords=("open", "dehisc", "gap", "split", "stitch",
                              "suture", "staple", "came apart", "loose",
                              "closed"),
        ),
        ProbeDimension(
            key="duration",
            question="Which day did you first notice this?",
            required=False,
            present_keywords=_DURATION_KEYWORDS,
            present_pattern=_DURATION_PATTERN,
        ),
    ),

    # ---- Generic fallback ----------------------------------------------
    # An unrecognised symptom still deserves the questions that are useful
    # for anything: how long, how bad, what else.
    SymptomCategory.UNKNOWN: (
        ProbeDimension(
            key="duration",
            question="How long has that been going on for?",
            required=True,
            present_keywords=_DURATION_KEYWORDS,
            present_pattern=_DURATION_PATTERN,
        ),
        ProbeDimension(
            key="functional_impact",
            question=(
                "Is it affecting your day-to-day - eating, sleeping, or "
                "getting about?"
            ),
            required=True,
            present_keywords=("affect", "sleep", "eating", "walking",
                              "work", "chores", "daily", "getting about",
                              "getting around", "stopping me", "can't"),
            present_pattern=_SEVERITY_PATTERN,
        ),
        ProbeDimension(
            key="associated",
            question="Apart from that, is anything else troubling you?",
            required=False,
            present_keywords=_ASSOCIATED_KEYWORDS,
        ),
        ProbeDimension(
            key="trend",
            question=(
                "Compared with when we last spoke, is it better, worse, or "
                "about the same?"
            ),
            required=False,
            present_keywords=("better", "worse", "same", "unchanged",
                              "improv", "no different"),
        ),
    ),
}


def probe_dimensions(category: SymptomCategory) -> tuple[ProbeDimension, ...]:
    """The checklist for a category, falling back to the generic one."""
    return PROBE_REGISTRY.get(category, PROBE_REGISTRY[SymptomCategory.UNKNOWN])


# ---------------------------------------------------------------------------
# Coverage detection
# ---------------------------------------------------------------------------

def _dimension_present(text: str, dimension: ProbeDimension) -> bool:
    """True when the patient already covered this dimension unprompted."""
    raw = text or ""
    pattern = _any_pattern(dimension.present_keywords)
    if pattern is not None and pattern.search(raw):
        return True
    if dimension.present_pattern and re.search(dimension.present_pattern, raw,
                                               re.IGNORECASE):
        return True
    return False


def is_non_symptom_answer(text: Any, category: Optional[SymptomCategory] = None) -> bool:
    """
    True when there is nothing to probe, because the answer is already
    complete rather than vague.

    Three cases:
      * Empty.
      * A bare scale answer ("6", "7/10") - precisely what was asked for.
      * A denial or "no change". The check is gated on the category being
        UNKNOWN so that a mixed answer ("no fever, but my arm hurts") is
        still probed for the part that matters. Positive self-reports
        ("doing well") only count when nothing negates them, so "not doing
        well" stays a complaint.

    Heuristic by nature. The extractor's needs_clarification flag is the
    authoritative signal - see classify_ambiguity().
    """
    raw = "" if text is None else str(text).strip()
    if not raw:
        return True
    if _BARE_SCALE_PATTERN.match(raw):
        return True

    cat = category if category is not None else classify_symptom(raw)
    if cat is not SymptomCategory.UNKNOWN:
        return False

    if _DENIAL_PATTERN.search(raw):
        return True
    if _POSITIVE_PATTERN.search(raw) and not _NEGATOR_PATTERN.search(raw):
        return True
    return False


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def build_probe_plan(
    field_name: str,
    text: Any,
    category: Optional[SymptomCategory] = None,
    max_questions: int = MAX_PROBE_QUESTIONS,
) -> SymptomProbePlan:
    """
    Build the follow-up questions for one symptom description.

    `questions` gets the missing required dimensions, capped at
    `max_questions`; anything the cap excluded still shows up in `missing`.
    Missing optional dimensions go to `optional_questions`, limited to
    whatever budget the required ones left unused.

    Returns a plan with no questions when the answer needs no probing
    (a denial, a bare pain score, or a fully detailed description).
    """
    raw = "" if text is None else str(text).strip()
    cat = category if category is not None else classify_symptom(raw)
    budget = max(0, max_questions)

    if is_non_symptom_answer(raw, cat):
        return SymptomProbePlan(
            field_name=field_name,
            raw_text=raw,
            category=cat,
            covered=[],
            missing=[],
            questions=[],
        )

    dimensions = probe_dimensions(cat)
    covered = [d.key for d in dimensions if _dimension_present(raw, d)]
    covered_set = set(covered)
    missing = [d for d in dimensions if d.key not in covered_set]

    required_gaps = [d for d in missing if d.required][:budget]
    spare = budget - len(required_gaps)
    optional_gaps = [d for d in missing if not d.required][:spare]

    return SymptomProbePlan(
        field_name=field_name,
        raw_text=raw,
        category=cat,
        covered=covered,
        missing=[d.key for d in missing],
        questions=[d.question for d in required_gaps],
        optional_questions=[d.question for d in optional_gaps],
    )


def is_symptom_incomplete(text: Any, category: Optional[SymptomCategory] = None) -> bool:
    """
    True when a symptom description is missing a *required* dimension.

    This is the text-based fallback for the "symptom described but
    incomplete" judgement. Prefer ExtractedClaim.needs_clarification when
    the extractor sets it.
    """
    raw = "" if text is None else str(text).strip()
    cat = category if category is not None else classify_symptom(raw)
    if is_non_symptom_answer(raw, cat):
        return False
    return any(
        d.required and not _dimension_present(raw, d)
        for d in probe_dimensions(cat)
    )


# ---------------------------------------------------------------------------
# The fork: which kind of ambiguity is this?
# ---------------------------------------------------------------------------

def classify_ambiguity(
    claim: Any,
    symptom_fields: frozenset[str] = SYMPTOM_FIELDS,
) -> AmbiguityKind:
    """
    Decide whether a claim needs confirmation, probing, or nothing.

    `claim` is duck-typed on triage.models.ExtractedClaim: it needs
    `field_name`, `confidence`, `extracted_value`, and optionally
    `needs_clarification` / `needs_hitl` / `evidence_source`.

    Resolution order matters. TRANSCRIPTION_UNCLEAR wins whenever the words
    themselves are in doubt - including tied candidates at otherwise high
    confidence, and "insufficient" claims where nothing was captured at all.
    Probing an unheard sentence for clinical detail is meaningless; the
    confirmation ladder has to settle the wording first. (For a claim with
    no value and no candidates the ladder immediately returns
    MANUAL_REVIEW, which is the right destination.)

    UNDERSPECIFIED is the opposite case: the wording is settled and the
    clinical content is thin. It fires when the extractor set
    needs_clarification, or - as a fallback for extractors that do not set
    it - when a symptom field is missing a required dimension.
    """
    confidence = float(getattr(claim, "confidence", 0.0) or 0.0)
    field_name = getattr(claim, "field_name", "")
    value = getattr(claim, "extracted_value", None)

    if confidence < CONFIRMATION_ACCEPT_THRESHOLD:
        return AmbiguityKind.TRANSCRIPTION_UNCLEAR

    # High confidence overall, but the extractor could not separate two
    # readings - still a wording problem, not a detail problem.
    if getattr(claim, "needs_hitl", False):
        return AmbiguityKind.TRANSCRIPTION_UNCLEAR

    category = classify_symptom("" if value is None else str(value))

    # Explicit extractor judgement takes precedence over our keyword
    # heuristics, on any field - an unrecognised complaint still gets the
    # generic checklist (how long, what impact, anything else), which is
    # useful for almost anything a patient might raise.
    #
    # The one exception is an answer that plainly has nothing to clarify:
    # probing a denial makes for a worse call than trusting the flag is
    # worth.
    if getattr(claim, "needs_clarification", False):
        if is_non_symptom_answer(value, category):
            return AmbiguityKind.NONE
        return AmbiguityKind.UNDERSPECIFIED

    # Fallback for extractors that do not set the flag: only free-text
    # symptom fields are second-guessed from their wording alone.
    if field_name in symptom_fields and is_symptom_incomplete(value, category):
        return AmbiguityKind.UNDERSPECIFIED

    return AmbiguityKind.NONE


def plan_for_claim(
    claim: Any,
    max_questions: int = MAX_PROBE_QUESTIONS,
) -> Optional[SymptomProbePlan]:
    """
    Probe plan for a claim, or None when probing is not the right tool.

    Returns None for anything that is not UNDERSPECIFIED - notably for
    claims that first need the confirmation ladder.
    """
    if classify_ambiguity(claim) is not AmbiguityKind.UNDERSPECIFIED:
        return None
    return build_probe_plan(
        field_name=getattr(claim, "field_name", ""),
        text=getattr(claim, "extracted_value", None),
        max_questions=max_questions,
    )
