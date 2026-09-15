"""Abstract base class shared by all six dimension checkers."""
from abc import ABC, abstractmethod

from arbitration.models import ArbitrationInput, DimensionName, DimensionResult


class BaseDimension(ABC):
    """
    Contract for a single gate dimension.

    Rules:
      - check() MUST be deterministic and side-effect free.
      - check() MUST NOT make network calls or call an LLM.
      - LLM work is complete before this layer runs.
    """

    @property
    @abstractmethod
    def name(self) -> DimensionName: ...

    @abstractmethod
    def check(self, inp: ArbitrationInput) -> DimensionResult: ...
