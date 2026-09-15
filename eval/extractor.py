"""
Extractor interface and mock implementation.

The extractor is responsible for turning a raw call transcript into
structured LLMEvidence. This module defines:

  AbstractExtractor  – interface that both mock and real implementations satisfy
  MockExtractor      – reads pre-baked evidence from the scenario dict (no API call)
  # RealExtractor    – (TODO) calls OpenAI to extract evidence from transcript

Switching from mock to real:
    harness = RunHarness(extractor=RealExtractor(api_key=...))
All downstream code (arbitration gate, metrics) is unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from arbitration.models import LLMEvidence


class AbstractExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        transcript: str,
        scenario: dict,
    ) -> tuple[list[LLMEvidence], dict]:
        """
        Extract structured evidence from a transcript.

        Args:
            transcript: Full verbatim call transcript.
            scenario:   Scenario dict from JSONL (may contain mock data or
                        be ignored by a real extractor).

        Returns:
            (llm_evidence, raw_llm_output)
        """

    @property
    def token_usage(self) -> dict:
        """Accumulated token usage. Overridden by API-backed extractors."""
        return {}


class MockExtractor(AbstractExtractor):
    """
    Returns pre-baked evidence stored in the scenario dict.

    Used during harness development before the real LLM extractor is wired up.
    Each scenario in test_scenarios.jsonl contains `mock_llm_evidence` and
    `mock_raw_llm_output` that this class reads verbatim.
    """

    def extract(
        self,
        transcript: str,
        scenario: dict,
    ) -> tuple[list[LLMEvidence], dict]:
        evidence = [
            LLMEvidence(
                field_name=e["field_name"],
                extracted_value=e["extracted_value"],
                citations=e.get("citations", []),
                confidence=float(e["confidence"]),
                reasoning=e.get("reasoning", ""),
            )
            for e in scenario.get("mock_llm_evidence", [])
        ]
        raw = scenario.get("mock_raw_llm_output", {})
        return evidence, raw
