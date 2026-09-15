"""
arbitration – Six-dimensional acceptance gate for FollowUp Companion call results.

Public surface:
    from arbitration import ArbitrationGate, ArbitrationInput, ArbitrationResult, Action
"""
from arbitration.gate import ArbitrationGate
from arbitration.models import (
    Action,
    ArbitrationInput,
    ArbitrationResult,
    CallSignals,
    DimensionName,
    DimensionResult,
    LLMEvidence,
)

__all__ = [
    "ArbitrationGate",
    "ArbitrationInput",
    "ArbitrationResult",
    "Action",
    "CallSignals",
    "DimensionName",
    "DimensionResult",
    "LLMEvidence",
]
