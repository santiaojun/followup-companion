"""
Test suite – Risk signal scenarios.

Covers two detection paths:
  Path 1 – Regex scan of the raw transcript (keyword matching).
  Path 2 – LLM high-confidence emergency_concerns (semantic path).

When either path fires the gate must:
  - Return TRIAGE_RISK regardless of whether quality dimensions pass.
  - Populate risk_flags with the specific signal categories detected.
  - Set priority = 1 (highest urgency).
"""
import copy

import pytest

from arbitration import Action, ArbitrationGate
from arbitration.dimensions.risk_signal import LLM_EMERGENCY_CONFIDENCE_THRESHOLD
from arbitration.models import ArbitrationInput, DimensionName, LLMEvidence
from arbitration.tests.conftest import (
    NORMAL_SIGNALS,
    NORMAL_EVIDENCE,
    build_normal_input,
)


@pytest.fixture
def gate() -> ArbitrationGate:
    return ArbitrationGate()


def _inject_transcript(transcript: str, call_id: str = "CALL-RISK-001") -> ArbitrationInput:
    """Build an input with a custom transcript but otherwise valid evidence."""
    inp = build_normal_input(call_id)
    inp.transcript = transcript
    return inp


# ---------------------------------------------------------------------------
# 1. Suicidal ideation
# ---------------------------------------------------------------------------

SUICIDAL_TRANSCRIPT = """
Agent: How have you been feeling since the surgery?
Patient: Honestly, I don't see the point anymore. I want to die. I have no reason to live.
Agent: I'm very sorry to hear that. Can you tell me more about how you are feeling?
Patient: I've been taking my medication, pain is about 4 out of 10, but nothing matters.
Agent: We're going to make sure you get the help you need right away.
Patient: Whatever. I don't care.
""".strip()


class TestSuicidalIdeation:
    def test_action_is_triage_risk(self, gate):
        inp = _inject_transcript(SUICIDAL_TRANSCRIPT, "CALL-RISK-SUI")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK

    def test_priority_is_one(self, gate):
        inp = _inject_transcript(SUICIDAL_TRANSCRIPT)
        result = gate.evaluate(inp)
        assert result.priority == 1

    def test_suicidal_flag_present(self, gate):
        inp = _inject_transcript(SUICIDAL_TRANSCRIPT)
        result = gate.evaluate(inp)
        critical_flags = [f for f in result.risk_flags if "suicidal_ideation" in f]
        assert len(critical_flags) >= 1

    def test_passed_is_false(self, gate):
        inp = _inject_transcript(SUICIDAL_TRANSCRIPT)
        result = gate.evaluate(inp)
        assert result.passed is False

    def test_risk_signal_dimension_has_flags(self, gate):
        inp = _inject_transcript(SUICIDAL_TRANSCRIPT)
        result = gate.evaluate(inp)
        risk_dim = result.dimensions[DimensionName.RISK_SIGNAL.value]
        assert not risk_dim.passed
        assert any("suicidal_ideation" in f for f in risk_dim.flags)


# ---------------------------------------------------------------------------
# 2. Acute emergency symptoms (chest pain)
# ---------------------------------------------------------------------------

CHEST_PAIN_TRANSCRIPT = """
Agent: This is the follow-up call for your recent discharge. How are you feeling?
Patient: Not good. I've been having chest pain since this morning. I can't breathe properly.
Agent: That sounds serious. Are you alone at home right now?
Patient: Yes, I took my medication but the chest pain is getting worse. Pain is 8 out of 10.
Agent: Please call emergency services immediately. We are flagging this for your care team.
Patient: Okay.
""".strip()


class TestChestPainEmergency:
    def test_action_is_triage_risk(self, gate):
        inp = _inject_transcript(CHEST_PAIN_TRANSCRIPT, "CALL-RISK-CP")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK

    def test_acute_emergency_flag_present(self, gate):
        inp = _inject_transcript(CHEST_PAIN_TRANSCRIPT)
        result = gate.evaluate(inp)
        emergency_flags = [f for f in result.risk_flags if "acute_emergency" in f]
        assert len(emergency_flags) >= 1

    def test_risk_signal_score_is_low(self, gate):
        inp = _inject_transcript(CHEST_PAIN_TRANSCRIPT)
        result = gate.evaluate(inp)
        risk_dim = result.dimensions[DimensionName.RISK_SIGNAL.value]
        assert risk_dim.score < 0.7


# ---------------------------------------------------------------------------
# 3. Medication overdose
# ---------------------------------------------------------------------------

OVERDOSE_TRANSCRIPT = """
Agent: Good morning. How are you doing today?
Patient: Not well. I was so anxious last night that I took too many pills.
  I think I took all my tablets, maybe 15 of them.
Agent: I need to make sure you are safe right now. Are you experiencing any symptoms?
Patient: I feel dizzy. My pain is okay, about 5 out of 10. But I'm scared.
Agent: We are connecting you with emergency services immediately.
""".strip()


class TestMedicationOverdose:
    def test_action_is_triage_risk(self, gate):
        inp = _inject_transcript(OVERDOSE_TRANSCRIPT, "CALL-RISK-OD")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK

    def test_overdose_flag_present(self, gate):
        inp = _inject_transcript(OVERDOSE_TRANSCRIPT)
        result = gate.evaluate(inp)
        od_flags = [f for f in result.risk_flags if "medication_overdose" in f]
        assert len(od_flags) >= 1


# ---------------------------------------------------------------------------
# 4. Risk signal overrides quality failures
#    (even a low-confidence call with missing fields still → TRIAGE_RISK, not TRIAGE_AMBIGUOUS)
# ---------------------------------------------------------------------------

LOW_QUALITY_RISK_TRANSCRIPT = """
A: Hello?
B: I want to die.
A: I'm sorry to hear that. Can you tell me more?
B: Nothing. Forget it.
""".strip()


class TestRiskOverridesQuality:
    def test_risk_beats_ambiguous(self, gate):
        """TRIAGE_RISK must win even when completeness / confidence also fail."""
        inp = ArbitrationInput(
            call_id="CALL-RISK-LQ",
            patient_id="PAT-999",
            transcript=LOW_QUALITY_RISK_TRANSCRIPT,
            llm_evidence=[],          # Zero evidence → completeness + confidence fail
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Expected TRIAGE_RISK but got {result.action}. "
            f"risk_flags={result.risk_flags}"
        )


# ---------------------------------------------------------------------------
# 5. Fall / injury (HIGH severity)
# ---------------------------------------------------------------------------

FALL_TRANSCRIPT = """
Agent: Good afternoon. How have you been doing since your procedure?
Patient: Not great. I fell down the stairs yesterday. I can't get up easily.
Agent: I'm sorry to hear that. Are you in pain?
Patient: Yes, pain is about 7 out of 10. I'm taking my medication but the fall really hurt me.
Agent: Have you seen a doctor about the fall?
Patient: Not yet. I'm scared it might be a fracture.
""".strip()


class TestFallInjury:
    def test_action_is_triage_risk(self, gate):
        inp = _inject_transcript(FALL_TRANSCRIPT, "CALL-RISK-FALL")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK

    def test_fall_flag_is_high_severity(self, gate):
        inp = _inject_transcript(FALL_TRANSCRIPT)
        result = gate.evaluate(inp)
        high_flags = [f for f in result.risk_flags if f.startswith("high:")]
        assert len(high_flags) >= 1


# ---------------------------------------------------------------------------
# 6. Path 2 – LLM high-confidence emergency_concerns
#
#    Transcript contains indirect / euphemistic risk language that regex
#    patterns cannot match.  The LLM extractor flags it; Path 2 must fire.
# ---------------------------------------------------------------------------

# Indirect suicidal ideation via "burden" language.  Zero regex matches.
INDIRECT_RISK_TRANSCRIPT = """
Agent: Hello, this is your weekly follow-up call. How have you been feeling?
Patient: I have been feeling very low. I don't know how to explain it.
Agent: Can you tell me more about that?
Patient: I just feel like everyone around me would be better off without me. I am just a burden.
Agent: I am concerned about what you are saying. Can you tell me more?
Patient: It would be easier for all of them if I wasn't around. I have been thinking about this a lot.
Agent: Are you taking your medication as prescribed?
Patient: I suppose so. Pain is about 4 out of 10.
Agent: We take what you are saying very seriously.
Patient: I just don't want to be a burden anymore.
""".strip()

# LLM correctly flags indirect ideation with high confidence.
_HIGH_CONFIDENCE_EMERGENCY = LLMEvidence(
    field_name="emergency_concerns",
    extracted_value="indirect suicidal ideation – burden statements, passive wish to not exist",
    citations=["everyone around me would be better off without me"],
    confidence=0.87,
    reasoning="Indirect suicidal ideation through burden language; no regex match but clinically significant.",
)

# LLM flags same transcript but confidence is below threshold.
_LOW_CONFIDENCE_EMERGENCY = LLMEvidence(
    field_name="emergency_concerns",
    extracted_value="possible indirect concern – patient sounds low",
    citations=[],
    confidence=LLM_EMERGENCY_CONFIDENCE_THRESHOLD - 0.05,
    reasoning="Uncertain interpretation.",
)

# LLM reports 'none' with high confidence (normal call).
_HIGH_CONFIDENCE_NONE = LLMEvidence(
    field_name="emergency_concerns",
    extracted_value="none",
    citations=["No, nothing urgent."],
    confidence=0.93,
    reasoning="Patient denied emergency concerns.",
)

# Variants that should also be treated as 'no emergency'.
_HIGH_CONFIDENCE_NO_CONCERNS = LLMEvidence(
    field_name="emergency_concerns",
    extracted_value="no urgent concerns",
    citations=[],
    confidence=0.90,
    reasoning="",
)


def _build_indirect_input(emergency_ev: LLMEvidence, call_id: str = "CALL-INDIRECT-001") -> ArbitrationInput:
    """Indirect-risk transcript + full normal evidence except emergency_concerns."""
    base_evidence = [
        ev for ev in NORMAL_EVIDENCE if ev.field_name != "emergency_concerns"
    ]
    return ArbitrationInput(
        call_id=call_id,
        patient_id="PAT-INDIRECT",
        transcript=INDIRECT_RISK_TRANSCRIPT,
        llm_evidence=base_evidence + [emergency_ev],
        call_signals=NORMAL_SIGNALS,
    )


class TestLLMPathEmergency:
    """Path 2: LLM high-confidence emergency_concerns → TRIAGE_RISK."""

    def test_high_confidence_non_none_triggers_triage_risk(self, gate):
        """Core behaviour: indirect transcript + high-confidence LLM flag → TRIAGE_RISK."""
        inp = _build_indirect_input(_HIGH_CONFIDENCE_EMERGENCY)
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Expected TRIAGE_RISK from LLM path but got {result.action}"
        )

    def test_llm_flag_present_in_risk_flags(self, gate):
        inp = _build_indirect_input(_HIGH_CONFIDENCE_EMERGENCY)
        result = gate.evaluate(inp)
        llm_flags = [f for f in result.risk_flags if f.startswith("llm:")]
        assert len(llm_flags) >= 1, f"Expected llm: flag; got {result.risk_flags}"

    def test_priority_is_one(self, gate):
        inp = _build_indirect_input(_HIGH_CONFIDENCE_EMERGENCY)
        result = gate.evaluate(inp)
        assert result.priority == 1

    def test_risk_dimension_score_is_below_one(self, gate):
        inp = _build_indirect_input(_HIGH_CONFIDENCE_EMERGENCY)
        result = gate.evaluate(inp)
        risk_dim = result.dimensions[DimensionName.RISK_SIGNAL.value]
        assert risk_dim.score < 1.0

    def test_low_confidence_does_not_trigger_path2(self, gate):
        """Below-threshold LLM confidence must NOT fire Path 2."""
        inp = _build_indirect_input(_LOW_CONFIDENCE_EMERGENCY)
        result = gate.evaluate(inp)
        # No regex match on indirect transcript, no high-confidence LLM flag →
        # quality gate may route to TRIAGE_AMBIGUOUS but not TRIAGE_RISK.
        assert result.action != Action.TRIAGE_RISK, (
            f"Low-confidence LLM evidence must not produce TRIAGE_RISK; got {result.action}"
        )
        llm_flags = [f for f in result.risk_flags if f.startswith("llm:")]
        assert llm_flags == []

    def test_none_value_with_high_confidence_does_not_trigger(self, gate):
        """'none' extracted_value must NOT fire Path 2 even at high confidence."""
        inp = _build_indirect_input(_HIGH_CONFIDENCE_NONE, "CALL-INDIRECT-NONE")
        result = gate.evaluate(inp)
        llm_flags = [f for f in result.risk_flags if f.startswith("llm:")]
        assert llm_flags == [], (
            f"High-confidence 'none' emergency_concerns must not produce llm: flag; "
            f"got {result.risk_flags}"
        )

    def test_no_urgent_concerns_variant_does_not_trigger(self, gate):
        """'no urgent concerns' variant must also be suppressed."""
        inp = _build_indirect_input(_HIGH_CONFIDENCE_NO_CONCERNS)
        result = gate.evaluate(inp)
        llm_flags = [f for f in result.risk_flags if f.startswith("llm:")]
        assert llm_flags == []

    # ------------------------------------------------------------------
    # Path 2 negation guard - whole-value, not prefix
    # ------------------------------------------------------------------
    # Regression coverage for a guard that used to test
    # ev_str.startswith("no ") / startswith("none"). That prefix test was
    # wrong in both directions, and an extractor prompt change surfaced it:
    # the model started phrasing "no concern" as "Not at all", which slipped
    # past the prefix and escalated stable patients, while a genuine "no
    # reason to live" WAS suppressed by it.

    @pytest.mark.parametrize("value", [
        "none", "No", "no", "NONE", "nothing", "nope", "nil",
        "Not at all", "Not at all.", "not really", "n/a",
        "no concerns", "no urgent concerns", "no issues", "no problems",
        "None reported", "None stated", "nothing new", "nothing else",
        "denies any concerns", "  none  ", "No.", "", None,
        # Intensifiers the model actually produces. "None whatsoever."
        # escaped an earlier version of this guard and escalated SC-022.
        "None whatsoever.", "nothing whatsoever", "No concerns at all.",
        "None at all", "no concerns to report", "None of note",
        "nothing so far", "No urgent concerns.",
        # Explicit clinical denials of self-harm. "No thoughts of ending my
        # life." is SC-025: the phrase contains the risk words and is a
        # denial, and it escalated 5/5 before this family was covered.
        "No thoughts of ending my life.", "no thoughts of hurting myself",
        "no thoughts of self harm", "No intention of ending my life",
        "no plans to hurt myself", "not suicidal", "no suicidal ideation",
        "denies suicidal ideation", "without any thoughts of self-harm",
        "no urges to hurt himself", "no thoughts of dying",
        # Compound denials. "No, nothing urgent at all." is what the live
        # model actually returns for SC-026, and the comma defeated a
        # single anchored match - every clause has to be checked.
        "No, nothing urgent at all.", "No, none whatsoever.",
        "No, nothing new", "None, no concerns", "no, not really",
        "No; nothing to report", "No - nothing urgent",
        "No, no thoughts of hurting myself.",
        # Leading first-person framing. "I have absolutely no thoughts of
        # ending my life." is what the live model returns for SC-025, and
        # the anchored match rejected it because of the three words in front.
        "I have absolutely no thoughts of ending my life.",
        "I genuinely have no thoughts of self-harm",
        "He has had no thoughts of self harm",
        "I have no concerns", "Patient reports no concerns",
        "Patient states no urgent concerns.", "She said no",
        "I am absolutely not suicidal", "absolutely none",
    ])
    def test_pure_denials_are_suppressed(self, gate, value):
        """A value that is only a denial must never fire Path 2."""
        ev = LLMEvidence(
            field_name="emergency_concerns",
            extracted_value=value,
            citations=["Patient reported no concerns."],
            confidence=0.95,
            reasoning="Patient denied concerns.",
        )
        inp = _build_indirect_input(ev, "CALL-DENIAL")
        result = gate.evaluate(inp)
        llm_flags = [f for f in result.risk_flags if f.startswith("llm:")]
        assert llm_flags == [], (
            f"Pure denial {value!r} must not produce an llm: flag; "
            f"got {result.risk_flags}"
        )

    @pytest.mark.parametrize("value", [
        # Begins with a negative word but carries a real concern. The old
        # prefix guard silently suppressed every one of these.
        "no reason to live",
        "no one to call and I am done",
        "nothing matters anymore",
        "none of this is worth it",
        "not able to breathe",
        "no point carrying on",
        "nothing to live for",
        "no hope left",
        "no future",
        "not worth living",
        # The pair that makes this a lexicon and not a prefix test: both
        # open with "no", and only one is a denial.
        #   "no thoughts of ending my life" -> suppressed (tested above)
        #   "no plans for the future"       -> hopelessness, must fire
        "no plans for the future",
        # Plain concerns, for contrast.
        "thoughts of letting go",
        "tired of fighting",
        "patient reports chest pain",
        "feeling like a burden",
        # Compound answers where one clause is a real concern. Suppressing
        # on "any clause denies" would silence these, so the rule is "every
        # clause must deny".
        "no, actually I have chest pain",
        "No, but I have been having chest pain",
        "none really, though I feel like a burden",
        "No, nothing urgent, but I have no reason to live",
        # First-person framing must not rescue a real concern: the denial
        # vocabulary after the prefix is what does the work, and "reason",
        # "hope" and "burden" are not in it.
        "I have no reason to live",
        "Patient reports no reason to live",
        "I have absolutely no hope left",
        "I am tired of fighting",
        "I feel like a burden",
        "I said no but I have chest pain",
        # Same nouns as the denials above, WITHOUT the negation.
        "I have thoughts of ending my life",
        "I have been having thoughts of self-harm",
    ])
    def test_concerns_beginning_with_a_negative_word_still_fire(self, gate, value):
        """
        A concern must not be suppressed just because it opens with "no".

        This is the safety-critical direction: Path 2 exists to catch the
        veiled phrasings Path 1's regex misses, so silencing it on a prefix
        match defeats its entire purpose.
        """
        ev = LLMEvidence(
            field_name="emergency_concerns",
            extracted_value=value,
            citations=[f"Patient said: {value}"],
            confidence=0.95,
            reasoning="Patient expressed a concern.",
        )
        inp = _build_indirect_input(ev, "CALL-VEILED")
        result = gate.evaluate(inp)
        assert "llm:high_confidence_emergency" in result.risk_flags, (
            f"Concern {value!r} must fire Path 2; got {result.risk_flags}"
        )
        assert result.action == Action.TRIAGE_RISK

    def test_normal_call_with_high_confidence_none_is_unaffected(self, gate):
        """A normal call (NORMAL_EVIDENCE has emergency_concerns='none' @ 0.90)
        must still produce ACCEPT, not TRIAGE_RISK."""
        inp = build_normal_input("CALL-NORMAL-PATH2")
        result = gate.evaluate(inp)
        assert result.action == Action.ACCEPT

    def test_both_paths_can_fire_simultaneously(self, gate):
        """Explicit suicidal transcript + high-confidence LLM flag → both kinds of flags."""
        inp = ArbitrationInput(
            call_id="CALL-BOTH-PATHS",
            patient_id="PAT-BOTH",
            transcript=SUICIDAL_TRANSCRIPT,   # regex triggers
            llm_evidence=(
                [ev for ev in NORMAL_EVIDENCE if ev.field_name != "emergency_concerns"]
                + [_HIGH_CONFIDENCE_EMERGENCY]  # LLM also triggers
            ),
            call_signals=NORMAL_SIGNALS,
        )
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK
        has_regex_flag = any(f.startswith("critical:") for f in result.risk_flags)
        has_llm_flag = any(f.startswith("llm:") for f in result.risk_flags)
        assert has_regex_flag and has_llm_flag, (
            f"Expected flags from both paths; got {result.risk_flags}"
        )


# ---------------------------------------------------------------------------
# 7. Behavioral farewell indicators (HIGH pattern, SC-017 fix)
# ---------------------------------------------------------------------------

FAREWELL_POSSESSIONS_TRANSCRIPT = """
Agent: How have you been keeping since we last spoke?
Patient: I've been sorting through things. I've been giving away some of my possessions,
  things I've had for years. My daughter asked why and I didn't really have an answer.
Agent: That sounds like a significant change. Can you tell me more about how you've been feeling?
Patient: I just feel like I want to say the things I never said. I've been writing letters
  to my grandchildren. Just to share things with them.
Agent: I want to make sure you're okay. Are you having any thoughts of harming yourself?
Patient: Not exactly. I just feel like everything is winding down.
Agent: I'm going to make sure your care team hears about this.
""".strip()

FAREWELL_AFFAIRS_TRANSCRIPT = """
Agent: Good afternoon. This is your weekly check-in call. How are things going?
Patient: I've been putting my affairs in order. Updating my will, contacting old friends.
Agent: Is there a particular reason you've been doing that now?
Patient: I just feel like it needs to be done. My pain level is about 5 out of 10.
Agent: I understand. Have you been having any dark thoughts lately?
Patient: Nothing specific. Just feels like the right time.
""".strip()


class TestBehavioralFarewell:
    """HIGH behavioral_farewell pattern must detect indirect suicidal signals."""

    def test_giving_away_possessions_triggers_triage_risk(self, gate):
        inp = _inject_transcript(FAREWELL_POSSESSIONS_TRANSCRIPT, "CALL-FAREWELL-001")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Expected TRIAGE_RISK for giving-away-possessions + farewell-letters; "
            f"got {result.action}, flags={result.risk_flags}"
        )

    def test_behavioral_farewell_flag_present(self, gate):
        inp = _inject_transcript(FAREWELL_POSSESSIONS_TRANSCRIPT)
        result = gate.evaluate(inp)
        farewell_flags = [f for f in result.risk_flags if "behavioral_farewell" in f]
        assert len(farewell_flags) >= 1, (
            f"Expected high:behavioral_farewell flag; got {result.risk_flags}"
        )

    def test_farewell_flag_is_high_severity(self, gate):
        inp = _inject_transcript(FAREWELL_POSSESSIONS_TRANSCRIPT)
        result = gate.evaluate(inp)
        high_farewell = [f for f in result.risk_flags if f == "high:behavioral_farewell"]
        assert len(high_farewell) >= 1

    def test_affairs_in_order_triggers_triage_risk(self, gate):
        inp = _inject_transcript(FAREWELL_AFFAIRS_TRANSCRIPT, "CALL-FAREWELL-002")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK

    def test_affairs_in_order_flag_present(self, gate):
        inp = _inject_transcript(FAREWELL_AFFAIRS_TRANSCRIPT)
        result = gate.evaluate(inp)
        farewell_flags = [f for f in result.risk_flags if "behavioral_farewell" in f]
        assert len(farewell_flags) >= 1

    def test_priority_is_one(self, gate):
        inp = _inject_transcript(FAREWELL_POSSESSIONS_TRANSCRIPT)
        result = gate.evaluate(inp)
        assert result.priority == 1

    def test_normal_winding_down_phrase_without_keywords_does_not_trigger(self, gate):
        """Generic 'winding down' language without farewell keywords must NOT trigger."""
        safe_transcript = """
Agent: How have you been?
Patient: I've been winding down at work, taking things easy since the surgery.
Agent: That sounds sensible. Pain level?
Patient: About 3 out of 10. I'm doing well overall.
""".strip()
        inp = _inject_transcript(safe_transcript, "CALL-FAREWELL-SAFE")
        result = gate.evaluate(inp)
        farewell_flags = [f for f in result.risk_flags if "behavioral_farewell" in f]
        assert farewell_flags == [], (
            f"'Winding down' without farewell keywords must not trigger; got {result.risk_flags}"
        )


# ---------------------------------------------------------------------------
# 8. Abuse minimization / victim fear language (HIGH pattern, SC-020 fix)
# ---------------------------------------------------------------------------

ABUSE_MINIMIZATION_TRANSCRIPT = """
Agent: How have you been feeling at home since the discharge?
Patient: Things have been difficult at home. My partner, he gets angry when I talk about
  my health. He doesn't mean anything by it, he's just under stress.
Agent: I'm concerned to hear that. Is there anything specific that worries you?
Patient: I'm actually a bit afraid to talk on the phone when he's around. I waited until
  he left the house to make this call.
Agent: Thank you for letting me know. Are you safe right now?
Patient: Yes. My pain is about 4 out of 10 and I've been taking my medication.
""".strip()

ABUSE_DIRECT_HIT_TRANSCRIPT = """
Agent: How are you doing?
Patient: Not great. He hit me last week. I'm scared.
Agent: I'm so sorry. Are you safe right now?
Patient: I think so. He's not home.
""".strip()


class TestAbuseMinimizerLanguage:
    """Victim minimization / fear language must trigger high:abuse_indicator."""

    def test_partner_anger_at_disclosure_triggers_triage_risk(self, gate):
        inp = _inject_transcript(ABUSE_MINIMIZATION_TRANSCRIPT, "CALL-ABUSE-001")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Expected TRIAGE_RISK for partner-anger-at-disclosure; "
            f"got {result.action}, flags={result.risk_flags}"
        )

    def test_abuse_indicator_flag_present(self, gate):
        inp = _inject_transcript(ABUSE_MINIMIZATION_TRANSCRIPT)
        result = gate.evaluate(inp)
        abuse_flags = [f for f in result.risk_flags if "abuse_indicator" in f]
        assert len(abuse_flags) >= 1, (
            f"Expected high:abuse_indicator flag; got {result.risk_flags}"
        )

    def test_minimization_flag_is_high_severity(self, gate):
        inp = _inject_transcript(ABUSE_MINIMIZATION_TRANSCRIPT)
        result = gate.evaluate(inp)
        high_abuse = [f for f in result.risk_flags if f == "high:abuse_indicator"]
        assert len(high_abuse) >= 1

    def test_afraid_to_talk_on_phone_triggers(self, gate):
        """'Afraid to talk on the phone' alone must trigger abuse_indicator."""
        transcript = """
Agent: Good morning. How are you?
Patient: Okay, I waited to call because I'm afraid to talk on the phone when
  my husband is around. But he's out now.
Agent: I see. Are you safe at home?
Patient: I think so. Pain is 3 out of 10.
""".strip()
        inp = _inject_transcript(transcript, "CALL-ABUSE-002")
        result = gate.evaluate(inp)
        abuse_flags = [f for f in result.risk_flags if "abuse_indicator" in f]
        assert len(abuse_flags) >= 1

    def test_existing_direct_hit_pattern_still_triggers(self, gate):
        """Original 'he hit me' pattern must not have been broken by the extension."""
        inp = _inject_transcript(ABUSE_DIRECT_HIT_TRANSCRIPT, "CALL-ABUSE-003")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK
        abuse_flags = [f for f in result.risk_flags if "abuse_indicator" in f]
        assert len(abuse_flags) >= 1


# ---------------------------------------------------------------------------
# 9. Context suppressors for CRITICAL patterns (SC-022–024 fix)
# ---------------------------------------------------------------------------

# SC-022: Educational overdose discussion — should NOT trigger.
OVERDOSE_EDUCATIONAL_TRANSCRIPT = """
Agent: I want to go over medication safety with you. It's important to understand
  that taking more than prescribed could lead to an overdose. The risk of overdose
  with this medication is something we take very seriously.
Patient: I understand. I've been careful about taking the correct dose.
Agent: Good. An overdose could cause serious harm. Are you storing the medication safely?
Patient: Yes, it's locked away. Pain is about 4 out of 10.
""".strip()

# SC-023: Resolved chest pain, cleared by cardiologist — should NOT trigger.
CHEST_PAIN_RESOLVED_TRANSCRIPT = """
Agent: How have you been since we last spoke?
Patient: Better now. I did have some chest pain about two weeks ago, but it went away
  after a day or so. I went to see my cardiologist and he said he's not worried,
  it was expected after the procedure and is completely normal.
Agent: That's reassuring. Any other concerns?
Patient: No, I feel much better. Pain level overall is about 2 out of 10.
""".strip()

# SC-024: Negated suicidal phrase — hyperbole qualifier — should NOT trigger.
HYPERBOLE_SUICIDAL_TRANSCRIPT = """
Agent: How was your week?
Patient: Oh, work was terrible. I felt like I had no reason to live for about
  ten minutes this morning when my computer crashed and I lost everything.
  But it was fine in the end, I recovered the files.
Agent: That sounds stressful! How are you feeling health-wise?
Patient: Much better actually. Pain is about 2 out of 10.
""".strip()

# Regression: genuine chest pain (unresolved, co-occurring symptoms) — MUST still trigger.
GENUINE_CHEST_PAIN_TRANSCRIPT = """
Agent: How are you feeling today?
Patient: Not well at all. I've had chest pain since yesterday and I can't breathe
  properly. I feel dizzy.
Agent: That sounds very serious. Please call emergency services.
Patient: I'm scared.
""".strip()

# Regression: genuine suicidal ideation — no hyperbole qualifier — MUST still trigger.
GENUINE_SUICIDAL_TRANSCRIPT = """
Agent: How have you been feeling?
Patient: Very low. I have no reason to live. I've been thinking about ending things.
Agent: I'm very concerned about what you're saying.
""".strip()

# Regression: genuine overdose statement — MUST still trigger.
GENUINE_OVERDOSE_TRANSCRIPT = """
Agent: Have you been taking your medication as prescribed?
Patient: No. I was so desperate last night, I took too many pills. I took all my tablets.
Agent: Are you feeling okay right now?
Patient: I feel very sick.
""".strip()


class TestContextSuppressors:
    """Path 1 suppressors must suppress false alarms without silencing genuine emergencies."""

    # ------------------------------------------------------------------
    # Suppressed cases (non-emergencies that previously fired)
    # ------------------------------------------------------------------

    def test_educational_overdose_discussion_suppressed(self, gate):
        """Educational discussion of overdose risk must NOT produce TRIAGE_RISK."""
        inp = _inject_transcript(OVERDOSE_EDUCATIONAL_TRANSCRIPT, "CALL-SUPPRESS-OD")
        result = gate.evaluate(inp)
        od_flags = [f for f in result.risk_flags if "medication_overdose" in f]
        assert od_flags == [], (
            f"Educational overdose context must be suppressed; got {result.risk_flags}"
        )
        assert result.action != Action.TRIAGE_RISK, (
            f"Expected non-TRIAGE_RISK for educational overdose; got {result.action}"
        )

    def test_resolved_cleared_chest_pain_suppressed(self, gate):
        """Chest pain resolved and cleared by cardiologist must NOT produce TRIAGE_RISK."""
        inp = _inject_transcript(CHEST_PAIN_RESOLVED_TRANSCRIPT, "CALL-SUPPRESS-CP")
        result = gate.evaluate(inp)
        emergency_flags = [f for f in result.risk_flags if "acute_emergency" in f]
        assert emergency_flags == [], (
            f"Resolved+cleared chest pain must be suppressed; got {result.risk_flags}"
        )
        assert result.action != Action.TRIAGE_RISK, (
            f"Expected non-TRIAGE_RISK for resolved chest pain; got {result.action}"
        )

    def test_time_bounded_hyperbole_suppressed(self, gate):
        """'No reason to live for ten minutes' hyperbole must NOT produce TRIAGE_RISK."""
        inp = _inject_transcript(HYPERBOLE_SUICIDAL_TRANSCRIPT, "CALL-SUPPRESS-SUI")
        result = gate.evaluate(inp)
        sui_flags = [f for f in result.risk_flags if "suicidal_ideation" in f]
        assert sui_flags == [], (
            f"Time-bounded hyperbole must be suppressed; got {result.risk_flags}"
        )
        assert result.action != Action.TRIAGE_RISK, (
            f"Expected non-TRIAGE_RISK for hyperbole; got {result.action}"
        )

    # ------------------------------------------------------------------
    # Regression: genuine emergencies must NOT be suppressed
    # ------------------------------------------------------------------

    def test_genuine_chest_pain_with_breathing_difficulty_not_suppressed(self, gate):
        """Chest pain + can't breathe → suppressor must NOT fire (other acute present)."""
        inp = _inject_transcript(GENUINE_CHEST_PAIN_TRANSCRIPT, "CALL-GENUINE-CP")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Genuine chest pain+breathing must produce TRIAGE_RISK; got {result.action}"
        )
        emergency_flags = [f for f in result.risk_flags if "acute_emergency" in f]
        assert len(emergency_flags) >= 1

    def test_genuine_suicidal_ideation_not_suppressed(self, gate):
        """Plain suicidal statement without time qualifier must still produce TRIAGE_RISK."""
        inp = _inject_transcript(GENUINE_SUICIDAL_TRANSCRIPT, "CALL-GENUINE-SUI")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Genuine suicidal ideation must produce TRIAGE_RISK; got {result.action}"
        )

    def test_genuine_overdose_statement_not_suppressed(self, gate):
        """'Took too many pills' must still produce TRIAGE_RISK."""
        inp = _inject_transcript(GENUINE_OVERDOSE_TRANSCRIPT, "CALL-GENUINE-OD")
        result = gate.evaluate(inp)
        assert result.action == Action.TRIAGE_RISK, (
            f"Genuine overdose must produce TRIAGE_RISK; got {result.action}"
        )

    def test_chest_pain_unresolved_not_suppressed(self, gate):
        """Chest pain that did NOT resolve must still fire even without other symptoms."""
        transcript = """
Agent: How are you feeling?
Patient: I've had chest pain for three days now and it won't go away.
Agent: Are you seeing a doctor about it?
Patient: Not yet.
""".strip()
        inp = _inject_transcript(transcript, "CALL-SUPPRESS-UNRESOLVED-CP")
        result = gate.evaluate(inp)
        emergency_flags = [f for f in result.risk_flags if "acute_emergency" in f]
        assert len(emergency_flags) >= 1, (
            f"Unresolved chest pain must not be suppressed; got {result.risk_flags}"
        )
