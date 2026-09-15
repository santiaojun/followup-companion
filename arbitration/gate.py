"""
ArbitrationGate – orchestrates all seven dimension checkers and emits a final verdict.

Decision waterfall (evaluated in strict priority order):
  1. PII boundary failure        → REJECT          (hard stop; never persist leaked data)
  2. Transcription quality low   → TRIAGE_AMBIGUOUS (early exit; downstream dims unreliable)
  3. Risk signal detected        → TRIAGE_RISK      (patient safety; priority 1)
  4. OOD detected                → REJECT           (call is not a valid medical follow-up)
  5. Quality failure(s)          → TRIAGE_AMBIGUOUS (completeness / traceability / confidence)
  6. All clear                   → ACCEPT

Rationale for transcription quality position (same tier as PII, before risk/OOD/quality):
  A bad transcript means completeness, traceability, confidence, and OOD checks are all
  reasoning from noisy input and cannot be trusted. We short-circuit them and send the
  call to human review immediately. The reviewer can listen to the original recording and
  catch any risk signals or PII issues that the ASR may have mangled.

The gate is stateless and can be instantiated once and reused across calls.
"""
from arbitration.dimensions import (
    CompletenessDimension,
    ConfidenceDimension,
    OODDimension,
    PIIBoundaryDimension,
    RiskSignalDimension,
    TraceabilityDimension,
    TranscriptionQualityDimension,
)
from arbitration.dimensions.base import BaseDimension
from arbitration.models import (
    Action,
    ArbitrationInput,
    ArbitrationResult,
    DimensionName,
    DimensionResult,
)

# Dimensions that participate in the "quality" gate (not OOD / risk / PII / transcription).
QUALITY_DIMENSIONS: set[str] = {
    DimensionName.COMPLETENESS,
    DimensionName.TRACEABILITY,
    DimensionName.CONFIDENCE,
}


class ArbitrationGate:
    """
    Runs all seven dimension checks and returns a single ArbitrationResult.

    Usage:
        gate = ArbitrationGate()
        result = gate.evaluate(inp)
        if result.action == Action.TRIAGE_RISK:
            triage_queue.push(result, priority=result.priority)
    """

    def __init__(self, dimensions: list[BaseDimension] | None = None) -> None:
        self._dimensions: list[BaseDimension] = dimensions or [
            PIIBoundaryDimension(),
            TranscriptionQualityDimension(),
            RiskSignalDimension(),
            OODDimension(),
            CompletenessDimension(),
            TraceabilityDimension(),
            ConfidenceDimension(),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, inp: ArbitrationInput) -> ArbitrationResult:
        """Run all dimensions and return the final arbitration result."""
        dim_results: dict[str, DimensionResult] = {}
        for dim in self._dimensions:
            dim_results[dim.name.value] = dim.check(inp)

        return self._decide(inp.call_id, dim_results)

    # ------------------------------------------------------------------
    # Private decision logic (mirrors the waterfall in the module docstring)
    # ------------------------------------------------------------------

    def _decide(
        self,
        call_id: str,
        results: dict[str, DimensionResult],
    ) -> ArbitrationResult:

        # ── Rule 1: PII boundary failure ──────────────────────────────
        # Hard reject regardless of anything else. Never persist leaked data.
        pii = results.get(DimensionName.PII_BOUNDARY.value)
        if pii is not None and not pii.passed:
            return ArbitrationResult(
                call_id=call_id,
                passed=False,
                action=Action.REJECT,
                priority=1,
                dimensions=results,
                risk_flags=[],
                pii_scrubbed=True,
                rejection_reason="PII leak detected in structured extracted fields.",
            )

        # ── Rule 2: Transcription quality failure (early exit) ────────
        # If the transcript is too noisy, no downstream dimension can be trusted.
        # Route to human review immediately so a clinician can audit the recording.
        tq = results.get(DimensionName.TRANSCRIPTION_QUALITY.value)
        if tq is not None and not tq.passed:
            return ArbitrationResult(
                call_id=call_id,
                passed=False,
                action=Action.TRIAGE_AMBIGUOUS,
                priority=2,
                dimensions=results,
                risk_flags=[],
                rejection_reason=(
                    f"Transcription quality too low; human must review recording. "
                    f"Flags: {tq.flags}"
                ),
            )

        # ── Rule 3: Risk signal ───────────────────────────────────────
        # Escalate immediately; quality issues are secondary to patient safety.
        risk = results.get(DimensionName.RISK_SIGNAL.value)
        risk_flags = risk.flags if risk is not None else []
        if risk_flags:
            return ArbitrationResult(
                call_id=call_id,
                passed=False,
                action=Action.TRIAGE_RISK,
                priority=1,
                dimensions=results,
                risk_flags=risk_flags,
            )

        # ── Rule 4: OOD failure ───────────────────────────────────────
        # The transcript is not a valid medical follow-up; reject outright.
        ood = results.get(DimensionName.OOD.value)
        if ood is not None and not ood.passed:
            return ArbitrationResult(
                call_id=call_id,
                passed=False,
                action=Action.REJECT,
                priority=2,
                dimensions=results,
                risk_flags=[],
                rejection_reason=f"OOD detected: {ood.reason}",
            )

        # ── Rule 5: Quality dimension failures ────────────────────────
        # Route to human review when completeness / traceability / confidence fail.
        quality_failures = [
            name
            for name, r in results.items()
            if name in {d.value for d in QUALITY_DIMENSIONS} and not r.passed
        ]
        if quality_failures:
            return ArbitrationResult(
                call_id=call_id,
                passed=False,
                action=Action.TRIAGE_AMBIGUOUS,
                priority=2,
                dimensions=results,
                risk_flags=[],
                rejection_reason=f"Quality gate failed on: {quality_failures}",
            )

        # ── Rule 6: All clear ─────────────────────────────────────────
        return ArbitrationResult(
            call_id=call_id,
            passed=True,
            action=Action.ACCEPT,
            priority=3,
            dimensions=results,
            risk_flags=[],
        )
