"""
Dimension 4 – Risk Signal Detection

Two independent detection paths.  Either path alone is sufficient to set risk flags
and cause the gate to route the call to TRIAGE_RISK.

  Path 1 – Regex scan of the raw transcript (safety net)
      Scans the verbatim transcript for explicit clinical risk keywords.
      Does NOT depend on the LLM extractor in any way.  This is intentional:
      the LLM may soften, paraphrase, or miss alarming statements; regex is a
      backstop that fires directly on what the patient said.

      Context suppressors (Path 1 CRITICAL patterns only):
        After a CRITICAL pattern matches, a secondary contextual check can
        suppress the flag if the surrounding text confirms a non-emergency
        context (e.g. educational discussion of overdose risk, time-bounded
        hyperbolic expression, resolved-and-medically-cleared chest pain).
        HIGH patterns are NOT suppressed — they are already lower-certainty.

  Path 2 – LLM high-confidence emergency_concerns (semantic path)
      Checks whether the extractor's emergency_concerns field carries a confidence
      score at or above LLM_EMERGENCY_CONFIDENCE_THRESHOLD *and* a non-null,
      non-"none" extracted_value.  This path catches indirect or euphemistic risk
      signals (burden language, behavioural indicators, veiled "letting go" phrasing)
      that regex patterns cannot match.

      Guard conditions for Path 2:
        • Confidence must be >= LLM_EMERGENCY_CONFIDENCE_THRESHOLD (0.80).
        • extracted_value must not be blank or a "no emergency" placeholder.
        • A single LLM flag may not override confirmed "no emergency" by the agent.
      These guards prevent high-confidence "none" extractions from spuriously
      triggering escalation.

Scoring:
  1.0 = no risk detected.  Decremented per flag:
    critical regex flag  → −0.40 each
    high regex flag      → −0.20 each
    llm path flag        → −0.35 each (slightly below critical: still uncertain)

Behaviour (different from other dimensions):
  - This dimension NEVER blocks ACCEPT by itself.
  - If *any* flags are found, the gate forces TRIAGE_RISK.
"""
import re

from arbitration.dimensions.base import BaseDimension
from arbitration.models import ArbitrationInput, DimensionName, DimensionResult

# ---------------------------------------------------------------------------
# Path 2 configuration
# ---------------------------------------------------------------------------

# Minimum LLM confidence on emergency_concerns to trigger Path 2.
# Set high enough to require strong model conviction; low enough to catch
# clear indirect signals (burden language, behavioural indicators, etc.).
LLM_EMERGENCY_CONFIDENCE_THRESHOLD = 0.80

# extracted_value strings that indicate NO emergency (normalised to lowercase).
# If the LLM reports one of these even with high confidence, Path 2 does NOT fire.
_NONE_INDICATORS: frozenset[str] = frozenset({
    "none", "no", "none stated", "none reported", "no concerns",
    "no emergency", "no urgent concerns", "no issues", "nothing",
    "not applicable", "n/a", "unknown",
})

# Whole-value negation check for Path 2.
#
# This deliberately matches the ENTIRE value (anchored), not a prefix. A
# prefix test is unsound in both directions:
#
#   False negative (dangerous): "no reason to live" starts with "no " and
#       would be suppressed as a non-emergency, silencing the one field whose
#       job is to catch veiled ideation. Only Path 1's regex would still see
#       it, and Path 2 exists precisely for the phrasings Path 1 misses.
#   False positive (noisy): "not at all" does not start with "none"/"no ",
#       so a prefix test lets it through and escalates a patient who just
#       said there is nothing wrong.
#
# Anchoring fixes both: a value that is *only* a denial is suppressed, and a
# value that merely *begins* with a negative word is not.

# A denial that can stand alone as the entire answer.
_DENIAL_CORE = r"""
      no | none | nope | nil | nothing | negative | n/?a | unknown
    | not \s+ (?: at \s+ all | applicable | really | that \s+ i \s+ know \s+ of )
    | no \s+ (?: concerns? | issues? | emergency | emergencies
               | urgent \s+ concerns? | problems? | complaints?
               | new \s+ concerns? )
    | none \s+ (?: stated | reported | mentioned | noted )
    | nothing \s+ (?: else | new | urgent | serious )
    | (?: patient \s+ )? den(?:ies|ied) (?: \s+ any )?
        (?: \s+ (?: concerns? | emergency | issues? ) )?
"""

# Intensifiers a patient tacks onto a denial. Without these, "None
# whatsoever." escaped the guard and escalated a stable patient (SC-022).
_DENIAL_TAIL = r"""
    (?: \s+ (?: whatsoever | at \s+ all | to \s+ report | of \s+ note
              | to \s+ speak \s+ of | so \s+ far | right \s+ now
              | at \s+ the \s+ moment | today | currently | thankfully
              | really | else | new ) )*
"""

# An explicit clinical denial of self-harm: "No thoughts of ending my life."
#
# The trailing object list is the safety anchor, and the reason this is a
# lexicon rather than a prefix test. Both of these open with "no":
#
#   "no thoughts of ending my life"  -> a denial       -> suppress
#   "no reason to live"              -> hopelessness   -> must fire
#
# Requiring <negated noun> + of/to + <self-harm object> separates them.
# "no reason to live", "no plans for the future", "nothing to live for" and
# "no one to call" all fail to match, so they still reach the reviewer.
_SELF_HARM_DENIAL = r"""
      (?: no | not | without | den(?:ies|ied) ) (?: \s+ any )? \s+
      (?: suicidal \s+ | self [-\s] harm \s+ )?
      (?: thoughts? | ideation | intent(?:ion)?s? | plans? | urges? | desires? ) \s+
      (?: of | to | about | towards? ) \s+
      (?: harm | hurt | kill | end | ending | suicide | dying | die | self | taking )
      .*
    | (?: no | not (?: \s+ feeling )? ) \s+ (?: suicidal | self [-\s] harm ) \b .*
    | den(?:ies|ied) \s+ (?: any \s+ )? (?: suicidal | self [-\s] harm | si \b ) .*
"""

# Optional first-person framing a model puts in front of a denial:
# "I have absolutely no thoughts of ending my life." Bounded to a reported-
# speech verb plus emphasis adverbs, NOT arbitrary text - the denial
# vocabulary after it is still what does the work, so "I have no reason to
# live" and "I am tired of fighting" match nothing and still fire.
# At most three tokens, each drawn from a whitelist of pronouns, auxiliary /
# reported-speech verbs and emphasis adverbs. None of them carry clinical
# content, and a denial from _DENIAL_CORE / _SELF_HARM_DENIAL is still
# required afterwards - which is why "I have no reason to live" and
# "Patient reports no reason to live" match nothing and still fire. The
# bound stops this from becoming "any prefix at all".
_DENIAL_LEAD = r"""
    (?:
        (?: i | patient | he | she | they | we
          | have | has | had | am | is | are | 've
          | reports? | reported | states? | stated
          | said | says? | confirms? | confirmed | feels?
          | absolutely | definitely | certainly | genuinely | honestly
          | really | completely | totally | truly | currently )
        \s+
    ){0,3}
"""

_NO_EMERGENCY_VALUE = re.compile(
    rf"""^\s*(?: {_DENIAL_LEAD} )(?:
          (?: {_DENIAL_CORE} ) (?: {_DENIAL_TAIL} )
        | (?: {_SELF_HARM_DENIAL} )
        )\s*[.!]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


# Clause separators. Patients answer in compound denials - "No, nothing
# urgent at all." - and a single anchored match cannot span the comma.
_CLAUSE_SPLIT = re.compile(r"[,;]|\s+[-–—]\s+")


def _is_no_emergency_value(raw) -> bool:
    """
    True when emergency_concerns holds a pure denial rather than a concern.

    Blank / null counts as a denial. Anything with actual content beyond a
    negative answer does NOT, even if it starts with a negative word - see
    _NO_EMERGENCY_VALUE for why that distinction is a safety property.

    Compound answers are split on clause boundaries and suppressed only when
    EVERY clause is a denial. That keeps both of these right:

        "No, nothing urgent at all."      -> both clauses deny  -> suppress
        "no, actually I have chest pain"  -> clause 2 is a concern -> fire

    Requiring all clauses (rather than any) is the safe direction: one
    concern anywhere in the answer reaches a reviewer.
    """
    text = str(raw or "").strip()
    if not text:
        return True
    if text.lower() in _NONE_INDICATORS:
        return True

    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text)]
    clauses = [c for c in clauses if c and c.strip(".!?")]
    if not clauses:
        return True
    return all(_NO_EMERGENCY_VALUE.match(c) for c in clauses)

# ---------------------------------------------------------------------------
# Path 1 – CRITICAL patterns
# ---------------------------------------------------------------------------

# (flag_tag, compiled_pattern)
CRITICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "suicidal_ideation",
        re.compile(
            r"\b(want to die|kill myself|end my life|suicidal|no reason to live|"
            r"don'?t want to (be here|live)|wish I was dead)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "acute_emergency",
        re.compile(
            r"\b(chest pain|can'?t breathe|difficulty breathing|"
            r"stroke|unconscious|collapsed|heart attack|seizure|"
            r"coughing up blood|vomiting blood)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "medication_overdose",
        re.compile(
            r"\b(took too many|too many pills?|overdose|overdosed|"
            r"took all (my|the) (pills?|tablets?|medication))\b",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Path 1 – HIGH patterns
# ---------------------------------------------------------------------------

HIGH_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "fall_injury",
        re.compile(
            r"\b(fell( down)?|fallen|can'?t get up|broken bone|fracture|"
            r"I (fell|tripped|slipped))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "severe_pain",
        re.compile(
            r"\b(unbearable pain|pain (is |feels? )?(a |like a )?(10|ten)|"
            r"excruciating|agonizing|worst pain)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "abuse_indicator",
        re.compile(
            r"\b("
            r"someone hurt me|being hurt|domestic|physically abused|"
            r"afraid of (him|her|them|my (partner|spouse|husband|wife))|"
            r"he hit me|she hit me|"
            # Victim fear / minimization language (SC-020 fix)
            r"(he|she) gets? angry when (I )?(talk|speak|call|mention|say)|"
            r"(he|she) doesn'?t? mean (anything|it) by (it|this)|"
            r"afraid to (talk|speak|call|mention) (about it|about this|on the phone)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "behavioral_farewell",
        re.compile(
            r"\b("
            # Giving away possessions (SC-017 fix)
            r"giving away (some of )?(my|their|all (my|the))?\s*"
            r"(things|belongings|possessions|jewellery|jewelry|clothes|books)|"
            # Farewell communications
            r"say(?:ing)? the things (I|they|we) never said|"
            r"writing (goodbye |farewell )?letters? to (say|tell|share)|"
            # Putting affairs in order
            r"putting (my |our |all my |all our )?affairs in order|"
            r"getting (my |our |all my |all our )?affairs in order"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Path 1 – Context suppressors for CRITICAL patterns
# ---------------------------------------------------------------------------
# These suppress a CRITICAL flag when surrounding text confirms a non-emergency
# context.  Suppressors are deliberately conservative: when in doubt, do NOT
# suppress.  HIGH patterns are not suppressed.

# medication_overdose suppressor:
# Fires when "overdose" appears in a clearly educational / informational context
# (e.g. patient asking about the risk of overdose, nurse explaining dangers).
_OVERDOSE_EDUCATIONAL = re.compile(
    r"\b(about|what).{0,30}\boverdose\b"
    r"|\b(importance|risk|danger|effects?|consequences?).{0,60}\boverdose\b"
    r"|\boverdose\b.{0,50}\b(could|might|can|would)\b",
    re.IGNORECASE,
)

# suicidal_ideation suppressor (narrow):
# Fires only when "no reason to live" is immediately followed by a time-bounded
# qualifier, indicating a transient hyperbolic expression
# (e.g. "no reason to live for about ten minutes" said as dark humour).
# Uses \s+ to handle the newlines that can appear between continuation lines.
_HYPERBOLE_TIME = re.compile(
    r"\bno reason to live\b\s+for\s+(about\s+)?\w+\s+(minutes?|seconds?|hours?|moments?|mins?)\b",
    re.IGNORECASE,
)

# acute_emergency (chest pain) suppressor:
# Chest pain alone in a post-discharge call requires ALL THREE of:
#   1. No co-occurring acute symptoms (can't breathe, stroke, seizure, etc.)
#   2. The pain episode is described as resolved / gone
#   3. A doctor/specialist has reviewed it and is not concerned
# Missing any one condition means the suppressor does NOT fire.
_CHEST_PAIN_ONLY = re.compile(r"\bchest pain\b", re.IGNORECASE)

_OTHER_ACUTE = re.compile(
    r"\b(can'?t breathe|difficulty breathing|stroke|unconscious|"
    r"collapsed|heart attack|seizure|coughing up blood|vomiting blood)\b",
    re.IGNORECASE,
)

_CHEST_PAIN_RESOLVED = re.compile(
    r"\b(went away|resolved|stopped|subsided|cleared up|eased|gone)\b",
    re.IGNORECASE,
)

_CHEST_PAIN_CLEARED = re.compile(
    r"\b(cardiologist|doctor|physician|specialist|gp|consultant)\b"
    r".{0,300}"
    r"\b(not concerned|not worried|expected|normal|okay|fine)\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_context_suppressed(tag: str, transcript: str) -> bool:
    """
    Return True if surrounding context confirms that a CRITICAL keyword match
    does NOT represent a real emergency and the flag should be dropped.

    Conservative by design: any ambiguity returns False (flag is kept).
    """
    if tag == "medication_overdose":
        return bool(_OVERDOSE_EDUCATIONAL.search(transcript))

    if tag == "suicidal_ideation":
        # Only suppress the narrow time-bounded hyperbole case.
        return bool(_HYPERBOLE_TIME.search(transcript))

    if tag == "acute_emergency":
        # Only suppress isolated chest pain that is both resolved AND cleared
        # by a clinician.  If any other acute symptom is present, never suppress.
        if not _CHEST_PAIN_ONLY.search(transcript):
            return False  # Something other than chest pain triggered — never suppress.
        if _OTHER_ACUTE.search(transcript):
            return False  # Co-occurring acute symptoms — keep the flag.
        return (
            bool(_CHEST_PAIN_RESOLVED.search(transcript))
            and bool(_CHEST_PAIN_CLEARED.search(transcript))
        )

    return False


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------

class RiskSignalDimension(BaseDimension):
    @property
    def name(self) -> DimensionName:
        return DimensionName.RISK_SIGNAL

    def check(self, inp: ArbitrationInput) -> DimensionResult:
        transcript = inp.transcript
        flags: list[str] = []

        # ── Path 1: regex scan of raw transcript ─────────────────────────
        for tag, pattern in CRITICAL_PATTERNS:
            if pattern.search(transcript):
                if _is_context_suppressed(tag, transcript):
                    continue  # Non-emergency context confirmed; skip this flag.
                flags.append(f"critical:{tag}")

        for tag, pattern in HIGH_PATTERNS:
            if pattern.search(transcript):
                flags.append(f"high:{tag}")

        # ── Path 2: LLM high-confidence emergency_concerns ────────────────
        # The LLM extractor has contextual understanding that keyword regex
        # cannot provide.  If it reports emergency_concerns with high confidence
        # *and* a non-null, non-"none" value, treat that as a risk signal.
        for ev in inp.llm_evidence:
            if ev.field_name == "emergency_concerns":
                if ev.confidence >= LLM_EMERGENCY_CONFIDENCE_THRESHOLD:
                    # Suppress the flag only when the whole value is a denial.
                    if not _is_no_emergency_value(ev.extracted_value):
                        flags.append("llm:high_confidence_emergency")
                break  # Only one emergency_concerns field expected per extraction

        # This dimension passes only when there are NO risk flags.
        # The gate's decision logic uses the flags directly to route to TRIAGE_RISK.
        passed = len(flags) == 0

        # Score: starts at 1.0, decremented per flag
        critical_count = sum(1 for f in flags if f.startswith("critical:"))
        high_count = sum(1 for f in flags if f.startswith("high:"))
        llm_count = sum(1 for f in flags if f.startswith("llm:"))
        score = max(0.0, 1.0 - 0.4 * critical_count - 0.2 * high_count - 0.35 * llm_count)

        reason = (
            "No clinical risk signals detected in transcript or LLM extraction."
            if passed
            else f"Risk signals found: {flags}. Requires immediate human review."
        )

        return DimensionResult(
            name=self.name,
            passed=passed,
            score=score,
            flags=flags,
            reason=reason,
        )
