"""
providers – CALL-E telephony client and three-tier degradation router.

Public surface:
    from providers import CalleClient, CallRouter, CallRequest, CallResult
"""
from providers.calle_client import CalleClient, CallRequest, CallResult
from providers.router import CallRouter

__all__ = ["CalleClient", "CallRequest", "CallResult", "CallRouter"]
