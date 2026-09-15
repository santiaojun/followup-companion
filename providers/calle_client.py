"""
CALL-E API client -- backed by the official calle-ai SDK.

Public interface is unchanged: CallRequest, CallResult, CalleAPIError,
and CalleClient are all still importable from this module.  Only the
internal HTTP implementation changes (urllib -> calle-ai / httpx).

Correct base URL:  https://api.heycall-e.com
Install SDK:       pip install calle-ai

The SDK uses a natural-language "task" string rather than a script-ID
field.  We construct the task from CallRequest.call_script_id and
patient_id so existing callers need no changes.  The call result's
transcript is requested via a JSON-Schema result_schema and extracted
from structured_result["transcript"] (falling back to evidence /
per-recipient transcript if needed).
"""
from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env before reading env vars (no-op if python-dotenv is not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Official SDK imports
# ---------------------------------------------------------------------------
from calle import CalleClient as _SDKClient                       # noqa: E402
from calle import (                                                # noqa: E402
    CalleAPIError         as _SDKAPIError,
    CalleAuthenticationError as _SDKAuthError,
    CalleConnectionError  as _SDKConnectionError,
    CalleRateLimitError   as _SDKRateLimitError,
    CalleTimeoutError     as _SDKTimeoutError,
)


# ---------------------------------------------------------------------------
# Public models  (interface unchanged from the urllib implementation)
# ---------------------------------------------------------------------------

@dataclass
class CallRequest:
    """Parameters for a single outbound follow-up call."""
    patient_id: str
    phone_number: str               # E.164 format, e.g. "+18005551234"
    call_script_id: str             # Used to construct the CALL-E task description
    scheduled_at_iso: Optional[str] = None   # ISO-8601; None = immediate
    metadata: dict = field(default_factory=dict)
    # BCP-47 language tag -- all patients on this deployment speak English.
    transcription_language: str = "en-US"
    # Optional pre-built task string. When set, replaces _TASK_TEMPLATE entirely.
    # Use this to inject CallPlanner questions + symptom probe rules into the
    # CALL-E conversation instructions.
    task_description: Optional[str] = None
    # Optional pre-built result_schema. When set, replaces _TRANSCRIPT_RESULT_SCHEMA.
    # Use this to request structured symptom-probe fields from CALL-E.
    result_schema: Optional[dict] = None


@dataclass
class CallResult:
    """
    Normalised call object returned by initiate_call and get_call_status.
    Wraps the raw SDK dict; callers should prefer named fields over .raw.
    """
    call_id: str
    status: str                     # "queued" | "in_progress" | "completed" | "failed" | "canceled"
    patient_id: str
    transcript: Optional[str] = None
    recording_url: Optional[str] = None
    duration_s: Optional[float] = None
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exception  (wraps SDK errors; preserves is_retryable for the router)
# ---------------------------------------------------------------------------

class CalleAPIError(Exception):
    """Raised when CALL-E returns an error or is unreachable."""

    def __init__(
        self,
        status_code: Optional[int],
        message: str,
        raw_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body

    @property
    def is_retryable(self) -> bool:
        """True for transient errors the router may retry (5xx, timeout, no connection)."""
        return self.status_code is None or self.status_code >= 500


# ---------------------------------------------------------------------------
# Internal: task builder and transcript result schema
# ---------------------------------------------------------------------------

_TASK_TEMPLATE = (
    "This is a post-care follow-up call for patient {patient_id}. "
    "Please conduct the call according to script template '{script_id}'. "
    "Cover the following topics: "
    "(1) any new or changed symptoms since the last appointment, "
    "(2) current medication adherence, "
    "(3) current pain level on a scale of 0 to 10, "
    "(4) whether they have attended their scheduled follow-up appointments, "
    "(5) any emergency or urgent concerns. "
    "Be warm, professional, and speak clearly in English."
)

# We ask CALL-E to return the verbatim transcript so our own extractor
# and arbitration gate can process the raw spoken text.
_TRANSCRIPT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["transcript"],
    "properties": {
        "transcript": {
            "type": "string",
            "description": (
                "The complete verbatim transcript of the call. "
                "Format each speaker turn as 'Speaker: utterance' on its own line "
                "using 'Agent:' and 'Patient:' as speaker labels. "
                "Mark inaudible segments as [unintelligible]."
            ),
        },
        "call_answered": {
            "type": "boolean",
            "description": "True if the patient answered the call.",
        },
        "duration_seconds": {
            "type": "number",
            "description": "Approximate call duration in seconds.",
        },
    },
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CalleClient:
    """
    CALL-E client backed by the official calle-ai SDK.

    Instantiate once; the SDK manages the underlying httpx connection pool.
    Use as a context manager to ensure the connection pool is closed cleanly.

        with CalleClient() as client:
            result = client.initiate_call(req)
    """

    DEFAULT_BASE_URL  = os.getenv("CALLE_BASE_URL", "https://api.heycall-e.com")
    DEFAULT_TIMEOUT_S = float(os.getenv("CALLE_TIMEOUT_S", "30"))

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        resolved_key = api_key or self._require_api_key()
        self.base_url  = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S
        self._sdk = _SDKClient(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

    def close(self) -> None:
        self._sdk.close()

    def __enter__(self) -> "CalleClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API  (interface unchanged)
    # ------------------------------------------------------------------

    def initiate_call(self, req: CallRequest) -> CallResult:
        """
        POST /v1/calls – queue an outbound call to the patient.

        Returns a CallResult with status="queued" on success.
        Raises CalleAPIError on failure.
        """
        task = req.task_description or _TASK_TEMPLATE.format(
            patient_id=req.patient_id,
            script_id=req.call_script_id,
        )

        locale = req.transcription_language or "en-US"
        # Derive CALL-E region from BCP-47 subtag (e.g. "en-US" -> "US")
        region = locale.split("-")[-1].upper() if "-" in locale else "US"

        # Deterministic idempotency key: prevents double-dialling on retry
        idempotency_key = (
            f"followup:{req.patient_id}:{req.call_script_id}"
            f":{req.scheduled_at_iso or _utcnow_iso()}"
        )

        metadata: dict[str, Any] = {
            "patient_id":   req.patient_id,
            "script_id":    req.call_script_id,
            **req.metadata,
        }

        logger.debug(
            "initiate_call patient=%s phone=%s script=%s idempotency_key=%s",
            req.patient_id, req.phone_number, req.call_script_id, idempotency_key,
        )

        schema = req.result_schema or _TRANSCRIPT_RESULT_SCHEMA

        try:
            raw = self._sdk.calls.create(
                task=task,
                recipients=[{
                    "phones":  [req.phone_number],
                    "region":  region,
                    "locale":  locale,
                }],
                result_schema=schema,
                metadata=metadata,
                idempotency_key=idempotency_key,
            )
        except (_SDKAPIError, _SDKAuthError, _SDKRateLimitError) as exc:
            raise CalleAPIError(
                status_code=exc.status_code,
                message=str(exc),
                raw_body=str(getattr(exc, "details", "")),
            ) from exc
        except (_SDKTimeoutError, _SDKConnectionError) as exc:
            raise CalleAPIError(status_code=None, message=str(exc)) from exc

        return self._parse_call_result(raw, patient_id=req.patient_id)

    def get_call_status(self, call_id: str) -> CallResult:
        """
        GET /v1/calls/{call_id} – retrieve current status and transcript (if complete).

        Raises CalleAPIError if call_id is not found (404) or on server error.
        """
        try:
            raw = self._sdk.calls.get(call_id)
        except (_SDKAPIError, _SDKAuthError, _SDKRateLimitError) as exc:
            raise CalleAPIError(
                status_code=exc.status_code,
                message=str(exc),
                raw_body=str(getattr(exc, "details", "")),
            ) from exc
        except (_SDKTimeoutError, _SDKConnectionError) as exc:
            raise CalleAPIError(status_code=None, message=str(exc)) from exc

        return self._parse_call_result(raw)

    def cancel_call(self, call_id: str) -> bool:
        """
        DELETE /v1/calls/{call_id} – cancel a queued call.

        Returns True on success, raises CalleAPIError otherwise.
        The SDK does not expose a dedicated cancel method; we call DELETE
        directly via the SDK's underlying httpx client.
        """
        try:
            path = f"/v1/calls/{quote(call_id, safe='').replace('.', '%2E')}"
            resp = self._sdk._client.request("DELETE", path)
            if resp.status_code >= 400:
                raise CalleAPIError(
                    status_code=resp.status_code,
                    message=f"Cancel failed: HTTP {resp.status_code} {resp.text[:200]}",
                )
        except CalleAPIError:
            raise
        except Exception as exc:
            raise CalleAPIError(status_code=None, message=str(exc)) from exc
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_call_result(raw: dict, patient_id: str = "") -> CallResult:
        """
        Convert a CALL-E SDK call-object dict into our CallResult model.

        Transcript extraction priority:
          1. structured_result["transcript"] -- requested via result_schema (preferred)
          2. top-level evidence string/list  -- raw evidence provided by some SDK versions
          3. recipients[0]["transcript"]     -- per-recipient fallback
        """
        call_id = str(raw.get("id") or raw.get("call_id", ""))
        status  = str(raw.get("status", "unknown"))

        # ── Transcript ─────────────────────────────────────────────────────
        transcript: Optional[str] = None

        structured = raw.get("structured_result") or {}
        if isinstance(structured, dict) and isinstance(structured.get("transcript"), str):
            transcript = structured["transcript"].strip() or None

        if not transcript:
            evidence = raw.get("evidence")
            if isinstance(evidence, str) and evidence.strip():
                transcript = evidence.strip()
            elif isinstance(evidence, list):
                lines: list[str] = []
                for item in evidence:
                    if isinstance(item, dict):
                        speaker = item.get("speaker", "Speaker")
                        text    = item.get("text",    "")
                        lines.append(f"{speaker}: {text}")
                    elif isinstance(item, str):
                        lines.append(item)
                if lines:
                    transcript = "\n".join(lines)

        if not transcript:
            for r in (raw.get("recipients") or []):
                if isinstance(r, dict):
                    t = r.get("transcript") or r.get("evidence")
                    if isinstance(t, str) and t.strip():
                        transcript = t.strip()
                        break

        # ── Duration ───────────────────────────────────────────────────────
        duration: Optional[float] = None
        for key in ("duration_s", "duration"):
            d = raw.get(key)
            if d is not None:
                try:
                    duration = float(d)
                    break
                except (TypeError, ValueError):
                    pass
        if duration is None and isinstance(structured, dict):
            ds = structured.get("duration_seconds")
            if ds is not None:
                try:
                    duration = float(ds)
                except (TypeError, ValueError):
                    pass

        # ── Patient ID ─────────────────────────────────────────────────────
        resolved_patient_id = patient_id or (
            (raw.get("metadata") or {}).get("patient_id", "")
        )

        return CallResult(
            call_id=call_id,
            status=status,
            patient_id=resolved_patient_id,
            transcript=transcript,
            recording_url=raw.get("recording_url"),
            duration_s=duration,
            raw=raw,
        )

    @staticmethod
    def _require_api_key() -> str:
        key = os.getenv("CALLE_API_KEY", "").strip()
        if not key:
            raise EnvironmentError(
                "CALLE_API_KEY is not set. "
                "Add it to your .env file or set it in the shell environment."
            )
        return key


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
