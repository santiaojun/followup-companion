"""
Three-tier degradation router for outbound calls.

Tier 1 – CALL-E primary:          Attempt the call via the live CALL-E API.
Tier 2 – Fallback provider:        If CALL-E is unavailable, try an alternate
                                   telephony provider (stub – fill in when ready).
Tier 3 – Graceful degradation:     If both providers fail, schedule a callback
                                   task for a human agent and emit a structured
                                   failure record so the triage queue is informed.

The router is synchronous and stateless. Retry delays and circuit-breaker state
are intentionally left to the caller (orchestrator layer) so this module stays
unit-testable without time-dependent logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from providers.calle_client import CalleAPIError, CalleClient, CallRequest, CallResult

logger = logging.getLogger(__name__)


@dataclass
class RouterOutcome:
    """Result of a routed call attempt, regardless of which tier succeeded."""
    success: bool
    tier_used: int                        # 1, 2, or 3
    call_result: Optional[CallResult]     # None on tier-3 (graceful degradation)
    failure_reason: Optional[str] = None  # Populated on tier-2/3 fallback
    callback_scheduled: bool = False      # True when tier-3 degradation fires


class CallRouter:
    """
    Routes an outbound call through up to three tiers of providers.

    Usage:
        router = CallRouter(primary=CalleClient())
        outcome = router.route(request)
        if not outcome.success:
            triage_queue.schedule_callback(request.patient_id)
    """

    def __init__(
        self,
        primary: CalleClient,
        fallback: Optional[object] = None,  # Replace with FallbackClient when available
    ) -> None:
        self._primary = primary
        self._fallback = fallback          # Optional[FallbackProviderClient]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, req: CallRequest) -> RouterOutcome:
        """Attempt the call through available tiers and return the outcome."""

        # ── Tier 1: CALL-E primary ────────────────────────────────────
        try:
            result = self._primary.initiate_call(req)
            logger.info(
                "Tier-1 success: call_id=%s patient_id=%s",
                result.call_id,
                req.patient_id,
            )
            return RouterOutcome(success=True, tier_used=1, call_result=result)
        except CalleAPIError as exc:
            logger.warning(
                "Tier-1 failed (patient_id=%s): %s (retryable=%s)",
                req.patient_id,
                exc,
                exc.is_retryable,
            )
            tier1_reason = str(exc)

        # ── Tier 2: Fallback provider ─────────────────────────────────
        if self._fallback is not None:
            try:
                result = self._call_fallback(req)
                logger.info(
                    "Tier-2 success via fallback provider: patient_id=%s",
                    req.patient_id,
                )
                return RouterOutcome(
                    success=True,
                    tier_used=2,
                    call_result=result,
                    failure_reason=f"Tier-1 failed: {tier1_reason}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Tier-2 fallback also failed (patient_id=%s): %s",
                    req.patient_id,
                    exc,
                )
                tier2_reason = str(exc)
        else:
            tier2_reason = "No fallback provider configured."

        # ── Tier 3: Graceful degradation ──────────────────────────────
        logger.error(
            "Tier-3 degradation: scheduling human callback for patient_id=%s. "
            "Tier-1: %s | Tier-2: %s",
            req.patient_id,
            tier1_reason,
            tier2_reason,
        )
        self._schedule_callback(req)

        return RouterOutcome(
            success=False,
            tier_used=3,
            call_result=None,
            failure_reason=f"All providers failed. Tier-1: {tier1_reason}; Tier-2: {tier2_reason}",
            callback_scheduled=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_fallback(self, req: CallRequest) -> CallResult:
        """
        Invoke the fallback telephony provider.

        STUB: replace this body with actual fallback client call when a
        secondary provider is available. The signature must return CallResult.
        """
        raise NotImplementedError(
            "Fallback provider is not yet implemented. "
            "Wire in a secondary telephony client here."
        )

    @staticmethod
    def _schedule_callback(req: CallRequest) -> None:
        """
        Emit a structured callback task when all providers are exhausted.

        STUB: integrate with the triage/ module's task queue here.
        For now, logs at ERROR level so on-call receives an alert.
        """
        logger.error(
            "CALLBACK_REQUIRED patient_id=%s phone=%s – "
            "human agent must attempt contact manually.",
            req.patient_id,
            req.phone_number,
        )
