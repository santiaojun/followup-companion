"""
Tests for symptom detail probing (the "heard it, still too vague" mechanism).

Headline scenarios
------------------
A - A general description triggers detail questions.
    "My arm hurts" is transcribed perfectly - confidence 0.97, no candidate
    ambiguity whatsoever - and is still unusable. It is classified as PAIN
    and probed for site first, then character and duration.

B - A specific description is not mistaken for a vague one.
    "Left inner forearm, constant dull ache, been three days now" covers
    every required dimension and must produce no probe at all.

C - The extractor's needs_clarification flag drives the trigger.
    A claim the extractor marked as incomplete is probed even when the
    keyword fallback would have let it through, and the confirmation ladder
    is still preferred whenever the wording itself is in doubt.

Coverage
--------
- Category routing for all four families, including the tie cases
  ("stomach pain" -> pain, "my incision hurts" -> wound)
- Dimensions the patient already volunteered are not re-asked
- A body region alone ("my arm") does not count as a specific site
- Vague time references ("recently") do not count as a duration answer
- Word-boundary matching: "sore" does not fire on "score"
- Denials ("no new symptoms") and bare scores ("6") are never probed
- Mixed answers ("no fever, but my arm hurts") are still probed
- Question budget: required gaps first, optional only with spare budget
"""
from __future__ import annotations

import pytest

from agents.models import (
    MAX_PROBE_QUESTIONS,
    AmbiguityKind,
    SymptomCategory,
)
from agents.symptom_probe import (
    PROBE_REGISTRY,
    build_probe_plan,
    classify_ambiguity,
    classify_symptom,
    is_non_symptom_answer,
    is_symptom_incomplete,
    plan_for_claim,
    probe_dimensions,
)
from triage.models import ExtractedClaim

VAGUE_PAIN = "my arm hurts"
SPECIFIC_PAIN = "left inner forearm, constant dull ache, been three days now"


def _claim(
    field_name: str = "symptom_update",
    value: str | None = VAGUE_PAIN,
    confidence: float = 0.95,
    candidates: list[dict] | None = None,
    evidence_source: str = "direct_quote",
    needs_clarification: bool = False,
) -> ExtractedClaim:
    return ExtractedClaim(
        field_name=field_name,
        evidence_source=evidence_source,
        extracted_value=value,
        candidates=candidates,
        citations=[value] if value else [],
        confidence=confidence,
        needs_clarification=needs_clarification,
    )


# ---------------------------------------------------------------------------
# Scenario A - a vague description is probed
# ---------------------------------------------------------------------------

class TestVagueSymptomIsProbed:

    def test_arm_pain_is_probed_for_site_character_and_duration(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN)

        assert plan.category is SymptomCategory.PAIN
        assert plan.has_questions
        # A region is not a site, so location is still open - and is asked
        # first, because it is first in the checklist.
        assert plan.missing[:3] == ["location", "quality", "duration"]
        assert len(plan.questions) == MAX_PROBE_QUESTIONS
        assert "Whereabouts exactly" in plan.questions[0]
        assert "What does it feel like" in plan.questions[1]
        assert "How long" in plan.questions[2]

    def test_arm_pain_is_clinically_incomplete(self):
        assert is_symptom_incomplete(VAGUE_PAIN)

    def test_location_probe_asks_for_laterality(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN)
        assert "left side or the right" in plan.questions[0]

    def test_exhausted_budget_leaves_no_room_for_optional(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN)
        assert plan.optional_questions == []
        # The un-asked optional gaps are still recorded.
        assert "severity" in plan.missing


# ---------------------------------------------------------------------------
# Scenario B - a specific description is left alone
# ---------------------------------------------------------------------------

class TestSpecificSymptomIsNotProbed:

    def test_specific_description_needs_no_required_probes(self):
        plan = build_probe_plan("symptom_update", SPECIFIC_PAIN)

        assert plan.category is SymptomCategory.PAIN
        assert set(plan.covered) >= {"location", "quality", "duration"}
        assert plan.questions == []

    def test_specific_description_is_not_flagged_incomplete(self):
        assert not is_symptom_incomplete(SPECIFIC_PAIN)

    def test_specific_description_produces_no_plan(self):
        assert plan_for_claim(_claim(value=SPECIFIC_PAIN, confidence=0.95)) is None

    def test_spare_budget_goes_to_optional_dimensions(self):
        plan = build_probe_plan("symptom_update", SPECIFIC_PAIN)
        # All required covered, so the optional gaps become available for a
        # caller that has room - but `questions` stays empty.
        assert plan.questions == []
        assert plan.optional_questions
        assert len(plan.optional_questions) <= MAX_PROBE_QUESTIONS


# ---------------------------------------------------------------------------
# Scenario C - the extractor flag is the authoritative trigger
# ---------------------------------------------------------------------------

class TestNeedsClarificationFlag:

    def test_flag_triggers_a_probe(self):
        claim = _claim(value="my shoulder is sore", confidence=0.95,
                       needs_clarification=True)

        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED
        plan = plan_for_claim(claim)
        assert plan is not None
        assert plan.category is SymptomCategory.PAIN
        assert plan.has_questions

    def test_flag_overrides_the_keyword_fallback(self):
        # Keyword-wise this looks complete (site, character, duration all
        # present), but the extractor read the whole transcript and
        # disagreed. The flag wins.
        claim = _claim(value=SPECIFIC_PAIN, confidence=0.95,
                       needs_clarification=True)

        assert not is_symptom_incomplete(SPECIFIC_PAIN)
        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED
        plan = plan_for_claim(claim)
        assert plan is not None
        # Nothing required is missing, so the optional gaps carry the probe.
        assert plan.optional_questions

    def test_flag_applies_to_non_symptom_fields_with_a_symptom_value(self):
        # "chest discomfort" under emergency_concerns absolutely deserves
        # the pain checklist.
        claim = _claim(field_name="emergency_concerns",
                       value="some chest discomfort", confidence=0.95,
                       needs_clarification=True)
        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED
        assert plan_for_claim(claim).category is SymptomCategory.PAIN

    def test_flag_is_ignored_when_there_is_nothing_to_clarify(self):
        # Probing a denial makes for a worse call than trusting the flag is
        # worth.
        claim = _claim(value="no new symptoms", confidence=0.95,
                       needs_clarification=True)
        assert classify_ambiguity(claim) is AmbiguityKind.NONE
        assert plan_for_claim(claim) is None

    def test_flag_does_not_override_transcription_doubt(self):
        # Both problems at once: settle the wording first, because you
        # cannot probe a sentence you did not hear.
        claim = _claim(value=VAGUE_PAIN, confidence=0.45,
                       needs_clarification=True)
        assert classify_ambiguity(claim) is AmbiguityKind.TRANSCRIPTION_UNCLEAR
        assert plan_for_claim(claim) is None

    def test_default_is_false_for_claims_that_do_not_set_it(self):
        assert _claim().needs_clarification is False

    def test_fallback_still_works_without_the_flag(self):
        claim = _claim(value=VAGUE_PAIN, confidence=0.97)
        assert claim.needs_clarification is False
        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED


# ---------------------------------------------------------------------------
# Category routing
# ---------------------------------------------------------------------------

class TestCategoryClassification:

    @pytest.mark.parametrize("text,expected", [
        ("my arm hurts", SymptomCategory.PAIN),
        ("a bit of pain in my chest", SymptomCategory.PAIN),
        ("my shoulder aches", SymptomCategory.PAIN),
        ("my knee is sore", SymptomCategory.PAIN),
        ("coughing a lot, bringing up phlegm", SymptomCategory.RESPIRATORY),
        ("I get short of breath walking", SymptomCategory.RESPIRATORY),
        ("a bit wheezy at night", SymptomCategory.RESPIRATORY),
        ("feeling nauseous and threw up twice", SymptomCategory.GASTROINTESTINAL),
        ("I've had diarrhea", SymptomCategory.GASTROINTESTINAL),
        ("no appetite and some heartburn", SymptomCategory.GASTROINTESTINAL),
        ("the wound looks a bit red", SymptomCategory.WOUND),
        ("something is weeping from the incision", SymptomCategory.WOUND),
        ("one of the stitches came out", SymptomCategory.WOUND),
    ])
    def test_families_route_to_their_checklist(self, text, expected):
        assert classify_symptom(text) == expected

    def test_stomach_pain_is_characterised_as_pain(self):
        # Both families match once; pain outranks GI on a tie because
        # site/character/duration is the more useful checklist here.
        assert classify_symptom("stomach pain") is SymptomCategory.PAIN

    def test_incision_pain_is_characterised_as_wound(self):
        # Wound outranks pain: infection signs are the time-critical ones.
        assert classify_symptom("my incision hurts") is SymptomCategory.WOUND

    def test_unrecognised_symptom_falls_back(self):
        assert classify_symptom("I keep waking up in the night") is (
            SymptomCategory.UNKNOWN
        )

    def test_empty_text_is_unknown(self):
        assert classify_symptom("") is SymptomCategory.UNKNOWN
        assert classify_symptom("   ") is SymptomCategory.UNKNOWN

    def test_unknown_category_still_gets_generic_questions(self):
        plan = build_probe_plan("symptom_update", "I just feel off")
        assert plan.category is SymptomCategory.UNKNOWN
        assert plan.has_questions
        assert "How long" in plan.questions[0]

    def test_word_boundary_matching_avoids_false_positives(self):
        # "score" must not match the pain keyword "sore".
        assert classify_symptom("my score was fine") is SymptomCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Not re-asking what the patient already said
# ---------------------------------------------------------------------------

class TestCoverageDetection:

    def test_partial_detail_only_probes_the_gaps(self):
        plan = build_probe_plan(
            "symptom_update", "sharp pain in my left knee for two days")

        assert set(plan.covered) >= {"location", "quality", "duration"}
        assert plan.questions == []

    def test_region_without_a_qualifier_is_not_a_site(self):
        # The point of the headline example: "my arm" names a region, not a site.
        assert "location" not in build_probe_plan("symptom_update", VAGUE_PAIN).covered
        assert "location" in build_probe_plan(
            "symptom_update", "my left arm hurts").covered

    def test_vague_time_reference_still_needs_a_duration_question(self):
        # "recently" is exactly the answer that needs following up, so it
        # must not count as covered.
        plan = build_probe_plan("symptom_update", "my left arm has been sore recently")
        assert "duration" in plan.missing
        assert any("How long" in q for q in plan.questions)

    def test_numeric_duration_is_detected_by_pattern(self):
        plan = build_probe_plan("symptom_update", "my shoulder has hurt for 3 days")
        assert "duration" in plan.covered

    def test_numeric_severity_is_detected_by_pattern(self):
        plan = build_probe_plan("symptom_update", "knee pain, about 6 out of 10")
        assert "severity" in plan.covered

    def test_gi_frequency_pattern(self):
        plan = build_probe_plan("symptom_update", "loose stools 5 times a day")
        assert plan.category is SymptomCategory.GASTROINTESTINAL
        assert "frequency" in plan.covered
        assert "stool_appearance" in plan.covered

    def test_wound_probes_infection_signs(self):
        plan = build_probe_plan(
            "symptom_update", "the wound is red and there is some fluid coming out")

        assert plan.category is SymptomCategory.WOUND
        assert "appearance" in plan.covered
        assert "discharge" in plan.covered
        assert "fever" in plan.missing
        assert any("fever" in q for q in plan.questions)

    def test_respiratory_probes_trigger_and_duration(self):
        plan = build_probe_plan("symptom_update", "I've been wheezy")

        assert plan.category is SymptomCategory.RESPIRATORY
        assert "trigger" in plan.missing
        assert "onset_duration" in plan.missing
        assert any("brings it on" in q for q in plan.questions)

    def test_respiratory_trigger_detected(self):
        plan = build_probe_plan(
            "symptom_update", "short of breath when I climb the stairs, started 4 days ago")
        assert "trigger" in plan.covered
        assert "onset_duration" in plan.covered
        assert plan.questions == []


# ---------------------------------------------------------------------------
# Answers that must never be probed
# ---------------------------------------------------------------------------

class TestNonSymptomAnswers:

    @pytest.mark.parametrize("text", [
        "no new symptoms", "nothing new", "no change since last time",
        "about the same", "none", "nothing at all", "no complaints",
        "everything is fine", "I'm fine", "doing well", "not really",
    ])
    def test_denials_are_not_probed(self, text):
        assert is_non_symptom_answer(text)
        assert not is_symptom_incomplete(text)
        assert build_probe_plan("symptom_update", text).questions == []

    @pytest.mark.parametrize("text", ["6", "6/10", "0", "10", "3 out of 10", "a 7"])
    def test_bare_scale_answers_are_complete(self, text):
        assert is_non_symptom_answer(text)
        assert build_probe_plan("pain_level", text).questions == []

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_values_are_not_probed(self, text):
        assert is_non_symptom_answer(text)
        assert build_probe_plan("symptom_update", text).questions == []

    def test_mixed_denial_and_symptom_is_still_probed(self):
        # A negation alongside a real complaint must not suppress the probe.
        text = "no fever, but my arm hurts"
        assert not is_non_symptom_answer(text)
        assert is_symptom_incomplete(text)
        assert build_probe_plan("symptom_update", text).has_questions

    def test_negated_positive_is_not_a_denial(self):
        # "doing well" ends the topic; "not doing well" does not.
        assert not is_non_symptom_answer("I'm not doing well at all")

    def test_improving_pain_is_still_a_symptom(self):
        assert not is_non_symptom_answer("my arm pain is a bit better")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestQuestionBudget:

    def test_default_cap(self):
        plan = build_probe_plan("symptom_update", "it hurts")
        assert len(plan.all_questions) <= MAX_PROBE_QUESTIONS

    @pytest.mark.parametrize("budget", [0, 1, 2, 5])
    def test_custom_budget_is_respected(self, budget):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN, max_questions=budget)
        assert len(plan.all_questions) <= budget

    def test_zero_budget_yields_no_questions_but_still_reports_gaps(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN, max_questions=0)
        assert plan.questions == []
        assert plan.missing  # the gaps are still known, just not asked

    def test_negative_budget_is_treated_as_zero(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN, max_questions=-3)
        assert plan.all_questions == []

    def test_explicit_category_overrides_detection(self):
        plan = build_probe_plan("symptom_update", VAGUE_PAIN,
                                category=SymptomCategory.WOUND)
        assert plan.category is SymptomCategory.WOUND
        assert any("draining" in q for q in plan.questions)


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_every_category_has_a_checklist(self):
        for category in SymptomCategory:
            assert probe_dimensions(category), category

    def test_every_category_has_at_least_one_required_dimension(self):
        for category, dims in PROBE_REGISTRY.items():
            assert any(d.required for d in dims), category

    def test_required_dimensions_fit_the_question_budget(self):
        # Otherwise a required gap could never be asked at all.
        for category, dims in PROBE_REGISTRY.items():
            required = [d for d in dims if d.required]
            assert len(required) <= MAX_PROBE_QUESTIONS, category

    def test_dimension_keys_are_unique_within_a_category(self):
        for category, dims in PROBE_REGISTRY.items():
            keys = [d.key for d in dims]
            assert len(keys) == len(set(keys)), category

    def test_every_dimension_has_a_question_and_a_detector(self):
        for category, dims in PROBE_REGISTRY.items():
            for d in dims:
                assert d.question.strip(), (category, d.key)
                assert d.present_keywords or d.present_pattern, (category, d.key)

    def test_pain_checklist_matches_the_documented_dimensions(self):
        keys = [d.key for d in PROBE_REGISTRY[SymptomCategory.PAIN]]
        assert keys[:4] == ["location", "quality", "duration", "associated"]

    def test_respiratory_checklist_requires_trigger_and_duration(self):
        required = [d.key for d in PROBE_REGISTRY[SymptomCategory.RESPIRATORY]
                    if d.required]
        assert required == ["trigger", "onset_duration"]

    def test_four_symptom_families_plus_fallback(self):
        assert set(PROBE_REGISTRY) == {
            SymptomCategory.PAIN,
            SymptomCategory.RESPIRATORY,
            SymptomCategory.GASTROINTESTINAL,
            SymptomCategory.WOUND,
            SymptomCategory.UNKNOWN,
        }


# ---------------------------------------------------------------------------
# The fork between the two kinds of ambiguity
# ---------------------------------------------------------------------------

class TestAmbiguityFork:

    def test_clear_but_vague_symptom_is_underspecified(self):
        claim = _claim(value=VAGUE_PAIN, confidence=0.97)
        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED

    def test_misheard_symptom_is_a_transcription_problem(self):
        # Same vague text, but we are not sure it is what was said. The
        # wording has to be settled before detail questions make sense.
        claim = _claim(value=VAGUE_PAIN, confidence=0.45)
        assert classify_ambiguity(claim) is AmbiguityKind.TRANSCRIPTION_UNCLEAR

    def test_tied_candidates_at_high_confidence_are_a_wording_problem(self):
        claim = _claim(
            value="stomach pain",
            confidence=0.90,
            candidates=[{"text": "stomach pain", "confidence": 0.90},
                        {"text": "chest pain", "confidence": 0.88}],
        )
        assert claim.needs_hitl
        assert classify_ambiguity(claim) is AmbiguityKind.TRANSCRIPTION_UNCLEAR

    def test_detailed_symptom_needs_nothing(self):
        claim = _claim(value=SPECIFIC_PAIN, confidence=0.95)
        assert classify_ambiguity(claim) is AmbiguityKind.NONE

    def test_denial_needs_nothing(self):
        claim = _claim(value="no new symptoms", confidence=0.95)
        assert classify_ambiguity(claim) is AmbiguityKind.NONE

    def test_non_symptom_field_is_never_underspecified_without_the_flag(self):
        claim = _claim(field_name="appointment_compliance", value="yes I went",
                       confidence=0.95)
        assert classify_ambiguity(claim) is AmbiguityKind.NONE

    def test_insufficient_evidence_routes_to_the_confirmation_path(self):
        # Nothing was captured, so there is no wording to probe. It lands on
        # the confirmation ladder, which immediately hands it to a human.
        claim = _claim(value=None, confidence=0.0, evidence_source="insufficient")
        assert classify_ambiguity(claim) is AmbiguityKind.TRANSCRIPTION_UNCLEAR

    def test_pain_level_is_a_symptom_field(self):
        claim = _claim(field_name="pain_level", value="it hurts quite a bit",
                       confidence=0.95)
        assert classify_ambiguity(claim) is AmbiguityKind.UNDERSPECIFIED

    def test_numeric_pain_level_needs_nothing(self):
        claim = _claim(field_name="pain_level", value="6", confidence=0.95)
        assert classify_ambiguity(claim) is AmbiguityKind.NONE


class TestPlanForClaim:

    def test_plan_is_built_for_underspecified_claims(self):
        plan = plan_for_claim(_claim(value=VAGUE_PAIN, confidence=0.97))
        assert plan is not None
        assert plan.field_name == "symptom_update"
        assert plan.category is SymptomCategory.PAIN
        assert plan.has_questions

    def test_no_plan_when_wording_is_in_doubt(self):
        assert plan_for_claim(_claim(value=VAGUE_PAIN, confidence=0.45)) is None

    def test_no_plan_when_nothing_is_needed(self):
        assert plan_for_claim(_claim(value="no new symptoms", confidence=0.95)) is None

    def test_budget_passes_through(self):
        plan = plan_for_claim(_claim(value=VAGUE_PAIN, confidence=0.97),
                              max_questions=1)
        assert plan is not None
        assert len(plan.all_questions) == 1
