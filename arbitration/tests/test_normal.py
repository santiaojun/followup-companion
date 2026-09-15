"""
Test suite – Normal / happy-path scenarios.

All six dimensions should pass and the gate should return ACCEPT.
"""
import pytest

from arbitration import Action, ArbitrationGate
from arbitration.models import DimensionName
from arbitration.tests.conftest import build_normal_input


@pytest.fixture
def gate() -> ArbitrationGate:
    return ArbitrationGate()


# ---------------------------------------------------------------------------
# 1. Full happy path
# ---------------------------------------------------------------------------

class TestFullNormalCall:
    def test_action_is_accept(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        assert result.action == Action.ACCEPT

    def test_passed_is_true(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        assert result.passed is True

    def test_no_risk_flags(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        assert result.risk_flags == []

    def test_pii_not_scrubbed(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        assert result.pii_scrubbed is False

    def test_all_seven_dimensions_present(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        expected = {d.value for d in DimensionName}
        assert expected == set(result.dimensions.keys())

    def test_all_dimensions_pass(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        for name, dim_result in result.dimensions.items():
            assert dim_result.passed, f"Dimension '{name}' unexpectedly failed: {dim_result.reason}"

    def test_priority_is_lowest(self, gate):
        """Accepted calls get priority 3 (routine); triage/risk would be 1 or 2."""
        inp = build_normal_input()
        result = gate.evaluate(inp)
        assert result.priority == 3


# ---------------------------------------------------------------------------
# 2. Multiple successive calls return consistent results (statelessness)
# ---------------------------------------------------------------------------

class TestGateIsStateless:
    def test_two_identical_calls_same_result(self, gate):
        inp = build_normal_input("CALL-A")
        r1 = gate.evaluate(inp)

        inp2 = build_normal_input("CALL-B")
        r2 = gate.evaluate(inp2)

        assert r1.action == r2.action
        assert r1.passed == r2.passed


# ---------------------------------------------------------------------------
# 3. Completeness dimension detail
# ---------------------------------------------------------------------------

class TestCompletenessOnNormalCall:
    def test_completeness_score_high(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        comp = result.dimensions[DimensionName.COMPLETENESS.value]
        assert comp.score >= 0.80

    def test_completeness_no_missing_field_flags(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        comp = result.dimensions[DimensionName.COMPLETENESS.value]
        missing = [f for f in comp.flags if f.startswith("missing_field:")]
        assert missing == []


# ---------------------------------------------------------------------------
# 4. Traceability dimension detail
# ---------------------------------------------------------------------------

class TestTraceabilityOnNormalCall:
    def test_traceability_score_high(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        assert trace.score >= 0.90

    def test_no_untraced_fields(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        trace = result.dimensions[DimensionName.TRACEABILITY.value]
        untraced = [f for f in trace.flags if f.startswith("untraced_field:")]
        assert untraced == []


# ---------------------------------------------------------------------------
# 5. Confidence dimension detail
# ---------------------------------------------------------------------------

class TestConfidenceOnNormalCall:
    def test_confidence_score_above_threshold(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        conf = result.dimensions[DimensionName.CONFIDENCE.value]
        assert conf.score >= 0.70

    def test_no_low_confidence_flags(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        conf = result.dimensions[DimensionName.CONFIDENCE.value]
        low_flags = [f for f in conf.flags if "low_confidence" in f]
        assert low_flags == []


# ---------------------------------------------------------------------------
# 6. OOD dimension detail
# ---------------------------------------------------------------------------

class TestOODOnNormalCall:
    def test_ood_passes(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        ood = result.dimensions[DimensionName.OOD.value]
        assert ood.passed is True

    def test_no_ood_flags(self, gate):
        inp = build_normal_input()
        result = gate.evaluate(inp)
        ood = result.dimensions[DimensionName.OOD.value]
        assert ood.flags == []
