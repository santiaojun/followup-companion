"""
Tests for CalleClient and CallRequest.

These tests do NOT make real HTTP calls.  They patch the official calle-ai
SDK's CalleCalls / CalleGoals resource objects so the entire network layer
is replaced by a controllable stub.

Covered:
  1. transcription_language defaults to "en-US" on CallRequest.
  2. initiate_call forwards locale and region correctly to the SDK.
  3. Callers can override transcription_language per request.
  4. initiate_call builds an idempotency_key containing patient_id and script_id.
  5. CalleAPIError.is_retryable is True for 5xx / connection errors.
  6. CalleAPIError.is_retryable is False for 4xx errors.
  7. get_call_status parses the returned call object correctly.
  8. Transcript extraction: structured_result > evidence string > recipients[0].
  9. CALLE_API_KEY missing raises EnvironmentError (not CalleAPIError).
 10. SDK CalleAPIError is translated to our CalleAPIError wrapper.
 11. SDK CalleConnectionError / CalleTimeoutError map to status_code=None.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.calle_client import CalleAPIError, CalleClient, CallRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(api_key: str = "test-key-123") -> CalleClient:
    return CalleClient(api_key=api_key, base_url="https://api.heycall-e.com")


def _sdk_call_dict(
    call_id: str = "call_001",
    status: str = "queued",
    transcript: str | None = None,
    duration_s: float | None = None,
) -> dict:
    """Minimal dict that the SDK's calls.create / calls.get would return."""
    d: dict = {"id": call_id, "status": status, "metadata": {"patient_id": "PAT-001"}}
    if transcript is not None:
        d["structured_result"] = {"transcript": transcript}
    if duration_s is not None:
        d["duration_s"] = duration_s
    return d


def _patch_sdk_create(mock_return: dict):
    """Patch _SDKClient so that calls.create() returns mock_return."""
    return patch(
        "providers.calle_client._SDKClient",
        return_value=_sdk_stub(create_return=mock_return),
    )


def _patch_sdk_get(mock_return: dict):
    return patch(
        "providers.calle_client._SDKClient",
        return_value=_sdk_stub(get_return=mock_return),
    )


def _sdk_stub(
    create_return: dict | None = None,
    get_return: dict | None = None,
) -> MagicMock:
    """Build a mock _SDKClient with controllable calls.create / calls.get."""
    stub = MagicMock()
    stub.calls.create.return_value = create_return or _sdk_call_dict()
    stub.calls.get.return_value    = get_return    or _sdk_call_dict()
    return stub


def _capture_create_kwargs(mock_sdk_cls) -> dict:
    """Extract kwargs passed to calls.create() from a patched _SDKClient."""
    instance = mock_sdk_cls.return_value
    return instance.calls.create.call_args.kwargs


# ---------------------------------------------------------------------------
# 1. CallRequest model defaults
# ---------------------------------------------------------------------------

class TestCallRequestDefaults:
    def test_transcription_language_defaults_to_en_us(self):
        req = CallRequest(
            patient_id="PAT-001",
            phone_number="+18005551234",
            call_script_id="SCRIPT-A",
        )
        assert req.transcription_language == "en-US"

    def test_other_fields_still_required(self):
        with pytest.raises(TypeError):
            CallRequest()   # missing required positional args

    def test_scheduled_at_defaults_to_none(self):
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        assert req.scheduled_at_iso is None

    def test_metadata_defaults_to_empty_dict(self):
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        assert req.metadata == {}

    def test_task_description_defaults_to_none(self):
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        assert req.task_description is None

    def test_result_schema_defaults_to_none(self):
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        assert req.result_schema is None


# ---------------------------------------------------------------------------
# 2. initiate_call: SDK payload forwarding
# ---------------------------------------------------------------------------

class TestInitiateCallPayload:
    @patch("providers.calle_client._SDKClient")
    def test_locale_forwarded_to_sdk(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(
            create_return=_sdk_call_dict()
        )
        client = _make_client()
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        recipients = kwargs["recipients"]
        assert len(recipients) == 1
        assert recipients[0]["locale"] == "en-US"

    @patch("providers.calle_client._SDKClient")
    def test_region_derived_from_en_us(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert kwargs["recipients"][0]["region"] == "US"

    @patch("providers.calle_client._SDKClient")
    def test_phone_forwarded_to_sdk(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert kwargs["recipients"][0]["phones"] == ["+18005551234"]

    @patch("providers.calle_client._SDKClient")
    def test_result_schema_requests_transcript(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        schema = kwargs.get("result_schema", {})
        assert "transcript" in schema.get("properties", {}), (
            "result_schema must request a 'transcript' property"
        )

    @patch("providers.calle_client._SDKClient")
    def test_task_contains_patient_id(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        client.initiate_call(CallRequest("PAT-XYZ", "+18005551234", "SCRIPT-A"))

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "PAT-XYZ" in kwargs["task"]

    @patch("providers.calle_client._SDKClient")
    def test_task_contains_script_id(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-FOLLOWUP-EN"))

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "SCRIPT-FOLLOWUP-EN" in kwargs["task"]

    @patch("providers.calle_client._SDKClient")
    def test_scheduled_at_absent_does_not_crash(self, mock_sdk_cls):
        """CallRequest with no scheduled_at_iso should still succeed."""
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        result = client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))
        assert result.call_id == "call_001"


# ---------------------------------------------------------------------------
# 3. transcription_language override
# ---------------------------------------------------------------------------

class TestTranscriptionLanguageOverride:
    @patch("providers.calle_client._SDKClient")
    def test_caller_can_override_language(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest(
            "PAT-001", "+18005551234", "SCRIPT-A",
            transcription_language="en-GB",
        )
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert kwargs["recipients"][0]["locale"] == "en-GB"
        assert kwargs["recipients"][0]["region"] == "GB"

    @patch("providers.calle_client._SDKClient")
    def test_override_does_not_affect_other_requests(self, mock_sdk_cls):
        """Each CallRequest is independent."""
        stub = _sdk_stub(create_return=_sdk_call_dict())
        mock_sdk_cls.return_value = stub
        client = _make_client()

        req_gb = CallRequest("PAT-001", "+18005551234", "SCRIPT-A", transcription_language="en-GB")
        req_us = CallRequest("PAT-002", "+18005559999", "SCRIPT-A")

        client.initiate_call(req_gb)
        args_gb = stub.calls.create.call_args.kwargs
        client.initiate_call(req_us)
        args_us = stub.calls.create.call_args.kwargs

        assert args_gb["recipients"][0]["locale"] == "en-GB"
        assert args_us["recipients"][0]["locale"] == "en-US"


# ---------------------------------------------------------------------------
# 3b. task_description and result_schema override
# ---------------------------------------------------------------------------

class TestTaskDescriptionOverride:
    @patch("providers.calle_client._SDKClient")
    def test_custom_task_description_replaces_template(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest(
            "PAT-001", "+18005551234", "SCRIPT-A",
            task_description="Custom task: call and ask about pain location.",
        )
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert kwargs["task"] == "Custom task: call and ask about pain location."

    @patch("providers.calle_client._SDKClient")
    def test_none_task_description_uses_template(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        # Template contains patient_id and script_id
        assert "PAT-001" in kwargs["task"]
        assert "SCRIPT-A" in kwargs["task"]

    @patch("providers.calle_client._SDKClient")
    def test_custom_result_schema_forwarded_to_sdk(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        custom_schema = {"type": "object", "properties": {"pain_location": {"type": "string"}}}
        client = _make_client()
        req = CallRequest(
            "PAT-001", "+18005551234", "SCRIPT-A",
            result_schema=custom_schema,
        )
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert kwargs["result_schema"] == custom_schema

    @patch("providers.calle_client._SDKClient")
    def test_none_result_schema_uses_default_transcript_schema(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest("PAT-001", "+18005551234", "SCRIPT-A")
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "transcript" in kwargs["result_schema"].get("properties", {})


# ---------------------------------------------------------------------------
# 4. Idempotency key
# ---------------------------------------------------------------------------

class TestIdempotencyKey:
    @patch("providers.calle_client._SDKClient")
    def test_idempotency_key_contains_patient_id(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "PAT-001" in kwargs.get("idempotency_key", "")

    @patch("providers.calle_client._SDKClient")
    def test_idempotency_key_contains_script_id(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-B"))

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "SCRIPT-B" in kwargs.get("idempotency_key", "")

    @patch("providers.calle_client._SDKClient")
    def test_scheduled_at_in_idempotency_key(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(create_return=_sdk_call_dict())
        client = _make_client()
        req = CallRequest(
            "PAT-001", "+18005551234", "SCRIPT-A",
            scheduled_at_iso="2026-09-15T09:00:00Z",
        )
        client.initiate_call(req)

        kwargs = _capture_create_kwargs(mock_sdk_cls)
        assert "2026-09-15T09:00:00Z" in kwargs.get("idempotency_key", "")


# ---------------------------------------------------------------------------
# 5. CalleAPIError retryability
# ---------------------------------------------------------------------------

class TestCalleAPIError:
    def test_5xx_is_retryable(self):
        err = CalleAPIError(status_code=503, message="Service unavailable")
        assert err.is_retryable is True

    def test_500_is_retryable(self):
        err = CalleAPIError(status_code=500, message="Internal server error")
        assert err.is_retryable is True

    def test_4xx_is_not_retryable(self):
        err = CalleAPIError(status_code=400, message="Bad request")
        assert err.is_retryable is False

    def test_404_is_not_retryable(self):
        err = CalleAPIError(status_code=404, message="Not found")
        assert err.is_retryable is False

    def test_connection_error_none_status_is_retryable(self):
        err = CalleAPIError(status_code=None, message="Connection refused")
        assert err.is_retryable is True

    @patch("providers.calle_client._SDKClient")
    def test_sdk_api_error_raises_our_calle_api_error(self, mock_sdk_cls):
        from calle import CalleAPIError as SDKError
        stub = MagicMock()
        stub.calls.create.side_effect = SDKError(
            code="internal_error",
            message="Something went wrong",
            status_code=503,
        )
        mock_sdk_cls.return_value = stub
        client = _make_client()
        with pytest.raises(CalleAPIError) as exc_info:
            client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))
        assert exc_info.value.status_code == 503
        assert exc_info.value.is_retryable is True

    @patch("providers.calle_client._SDKClient")
    def test_sdk_connection_error_maps_to_status_none(self, mock_sdk_cls):
        from calle import CalleConnectionError as SDKConnErr
        stub = MagicMock()
        stub.calls.create.side_effect = SDKConnErr("Connection refused")
        mock_sdk_cls.return_value = stub
        client = _make_client()
        with pytest.raises(CalleAPIError) as exc_info:
            client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))
        assert exc_info.value.status_code is None
        assert exc_info.value.is_retryable is True

    @patch("providers.calle_client._SDKClient")
    def test_sdk_timeout_error_maps_to_status_none(self, mock_sdk_cls):
        from calle import CalleTimeoutError as SDKTimeout
        stub = MagicMock()
        stub.calls.create.side_effect = SDKTimeout("Request timed out")
        mock_sdk_cls.return_value = stub
        client = _make_client()
        with pytest.raises(CalleAPIError) as exc_info:
            client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))
        assert exc_info.value.status_code is None
        assert exc_info.value.is_retryable is True

    @patch("providers.calle_client._SDKClient")
    def test_sdk_auth_error_raises_our_calle_api_error(self, mock_sdk_cls):
        from calle import CalleAuthenticationError as SDKAuthErr
        stub = MagicMock()
        stub.calls.create.side_effect = SDKAuthErr(
            code="unauthorized",
            message="Invalid API key",
            status_code=401,
        )
        mock_sdk_cls.return_value = stub
        client = _make_client()
        with pytest.raises(CalleAPIError) as exc_info:
            client.initiate_call(CallRequest("PAT-001", "+18005551234", "SCRIPT-A"))
        assert exc_info.value.status_code == 401
        assert exc_info.value.is_retryable is False


# ---------------------------------------------------------------------------
# 6. get_call_status – parsing and transcript extraction
# ---------------------------------------------------------------------------

class TestGetCallStatus:
    @patch("providers.calle_client._SDKClient")
    def test_returns_correct_call_id(self, mock_sdk_cls):
        mock_sdk_cls.return_value = _sdk_stub(
            get_return=_sdk_call_dict("call_XYZ", "completed", "Patient said they feel fine.")
        )
        result = _make_client().get_call_status("call_XYZ")
        assert result.call_id == "call_XYZ"
        assert result.status == "completed"

    @patch("providers.calle_client._SDKClient")
    def test_transcript_from_structured_result(self, mock_sdk_cls):
        raw = {
            "id": "call_001",
            "status": "completed",
            "structured_result": {"transcript": "Agent: Hello.\nPatient: Hi."},
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_001")
        assert result.transcript == "Agent: Hello.\nPatient: Hi."

    @patch("providers.calle_client._SDKClient")
    def test_transcript_fallback_to_evidence_string(self, mock_sdk_cls):
        raw = {
            "id": "call_002",
            "status": "completed",
            "evidence": "Agent: How are you?\nPatient: Fine.",
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_002")
        assert result.transcript == "Agent: How are you?\nPatient: Fine."

    @patch("providers.calle_client._SDKClient")
    def test_transcript_fallback_to_evidence_list(self, mock_sdk_cls):
        raw = {
            "id": "call_003",
            "status": "completed",
            "evidence": [
                {"speaker": "Agent",   "text": "How are you?"},
                {"speaker": "Patient", "text": "Fine."},
            ],
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_003")
        assert result.transcript == "Agent: How are you?\nPatient: Fine."

    @patch("providers.calle_client._SDKClient")
    def test_transcript_fallback_to_recipient(self, mock_sdk_cls):
        raw = {
            "id": "call_004",
            "status": "completed",
            "recipients": [{"transcript": "Agent: Hi.\nPatient: Hello."}],
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_004")
        assert result.transcript == "Agent: Hi.\nPatient: Hello."

    @patch("providers.calle_client._SDKClient")
    def test_structured_result_wins_over_evidence(self, mock_sdk_cls):
        raw = {
            "id": "call_005",
            "status": "completed",
            "structured_result": {"transcript": "preferred"},
            "evidence": "fallback",
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_005")
        assert result.transcript == "preferred"

    @patch("providers.calle_client._SDKClient")
    def test_duration_from_top_level_field(self, mock_sdk_cls):
        raw = _sdk_call_dict("call_001", "completed", duration_s=142.0)
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_001")
        assert result.duration_s == 142.0

    @patch("providers.calle_client._SDKClient")
    def test_duration_from_structured_result_schema(self, mock_sdk_cls):
        raw = {
            "id": "call_006",
            "status": "completed",
            "structured_result": {
                "transcript": "...",
                "duration_seconds": 200.0,
            },
        }
        mock_sdk_cls.return_value = _sdk_stub(get_return=raw)
        result = _make_client().get_call_status("call_006")
        assert result.duration_s == 200.0

    @patch("providers.calle_client._SDKClient")
    def test_get_call_status_passes_call_id_to_sdk(self, mock_sdk_cls):
        stub = _sdk_stub(get_return=_sdk_call_dict("call_ABC"))
        mock_sdk_cls.return_value = stub
        _make_client().get_call_status("call_ABC")
        stub.calls.get.assert_called_once_with("call_ABC")


# ---------------------------------------------------------------------------
# 7. Missing API key
# ---------------------------------------------------------------------------

class TestMissingAPIKey:
    def test_missing_key_raises_environment_error(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CALLE_API_KEY", None)
            with pytest.raises(EnvironmentError, match="CALLE_API_KEY"):
                CalleClient()
