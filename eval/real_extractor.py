"""
RealExtractor – OpenAI-backed clinical data extractor.

Three-stage extraction (Second Voice pattern):

  Stage 1 – Pure transcript analysis
      Ask the model to extract each required field directly from what the
      patient said.  No prior knowledge injected.  Output includes multiple
      candidates with confidence scores when the patient was unclear.

  Stage 2 – Portrait-assisted disambiguation  (stub)
      When candidates are too close in confidence (gap < CANDIDATE_CLOSENESS_THRESHOLD),
      the call would normally be sent to the patient's communication portrait
      (memory/) for disambiguation.  That module is not yet wired up, so this
      stage currently returns the Stage 1 fields unchanged.  The hook is here
      so the interface is stable when memory/ is ready.

  Stage 3 – Output normalisation
      Convert the model's JSON into LLMEvidence objects that the arbitration
      gate consumes.  Platform metadata (ASR confidence, etc.) that lives in
      scenario["mock_raw_llm_output"] is passed through unchanged – that data
      comes from the telephony platform, not from the LLM.

      The same fields are also normalised into ExtractedClaim objects,
      available via `last_claims` / `extract_claims()`.  Two views of one
      extraction, for two different consumers:

        LLMEvidence     -> arbitration gate (quality checks, risk detection)
        ExtractedClaim  -> triage queue + agents/ (HITL routing, and the
                           needs_clarification flag that drives symptom
                           detail probing)

  Stage 4 – Clarification judgement  (separate API call, skipped when unneeded)
      Decides needs_clarification for the free-text symptom fields: was the
      patient heard clearly but described something too general to act on?
      Deliberately independent of `confidence` — confidence is about the
      words, this is about whether the words are specific enough to use.

      This runs as its own call rather than as a sixth key in the Stage 1
      schema. Asking for it inline measurably degraded the safety-critical
      field: on SC-019 (veiled suicidal ideation) emergency_concerns came
      back as "None" in 5/10 runs where the unmodified prompt returned the
      concern 10/10, taking stable risk detection from 5/5 to 4/5. Keeping
      Stage 1's prompt byte-identical to the version the risk baseline was
      measured on removes that coupling entirely.

      Local filters skip the call when nothing could need clarifying (no
      symptom field, insufficient evidence, or a denial / precise score), so
      the extra cost only applies to calls that actually have a vague
      symptom in them. A failure in this stage is swallowed and degrades to
      needs_clarification=False; it can never break extraction.

      See agents/symptom_probe.py for what consumes the flag.

Model: gpt-4o-mini  (cost-effective; sufficient for structured extraction)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env before reading env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit(
        "openai package is not installed. Run: pip install openai"
    )

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbitration.models import LLMEvidence
from eval.extractor import AbstractExtractor
from triage.models import CANDIDATE_CLOSENESS_THRESHOLD, EVIDENCE_SOURCES, ExtractedClaim

# Imported rather than duplicated so the "nothing to clarify" vocabulary and
# the set of free-text symptom fields exist in exactly one place. The
# dependency runs eval -> agents only; agents deliberately does not import
# eval (see the note on STANDARD_PROTOCOL_ITEMS in agents/models.py), so
# there is no cycle.
from agents.models import SYMPTOM_FIELDS
from agents.symptom_probe import is_non_symptom_answer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0   # Deterministic – extraction should not be creative

# Model for the Stage 4 clarification judgement.  Separate constant so it can
# be downgraded independently: this is a single yes/no per symptom field and
# does not need the extraction model's headroom.
CLARIFICATION_MODEL = "gpt-4o-mini"

REQUIRED_FIELDS = [
    "symptom_update",
    "medication_adherence",
    "pain_level",
    "appointment_compliance",
    "emergency_concerns",
]

SYSTEM_PROMPT = f"""You are a clinical data extractor for a medical telephone follow-up system.
Extract structured information from a patient call transcript.

TASK
Return a JSON object with a "fields" array. It must contain exactly these 5 entries:
  symptom_update         – new or changed symptoms since last contact
  medication_adherence   – whether patient takes medications as prescribed
  pain_level             – current pain rating; prefer numeric 0-10 if stated
  appointment_compliance – whether patient attended recent scheduled appointments
  emergency_concerns     – any urgent concerns (suicidal ideation, acute pain, overdose, falls, abuse)

Each entry follows this schema:
{{
  "field_name": "<one of the 5 names above>",
  "evidence_source": "direct_quote" | "inferred" | "insufficient",
  "extracted_value": "<best single answer, or null if insufficient>",
  "candidates": [{{"text": "<interpretation>", "confidence": <0.0–1.0>}}, ...],
  "citations": ["<verbatim quote from transcript>"],
  "confidence": <0.0–1.0>,
  "reasoning": "<one-sentence explanation>"
}}

EVIDENCE SOURCE RULES
  direct_quote  – patient stated it explicitly in their own words
  inferred      – not stated directly but clearly implied by context
  insufficient  – transcript does not contain enough information

CANDIDATE RULES
  • Always provide at least one candidate.
  • If you have one clear best interpretation (confidence gap > 0.15 to next-best): one candidate.
  • If two or more interpretations are equally plausible (gap ≤ 0.15): list ALL, sorted descending.
  • Confidence is a float in [0.0, 1.0].

CITATION RULES
  • Citations MUST be verbatim quotes copied from the transcript.
  • For "insufficient" fields, citations may be empty.

SAFETY RULE – EMERGENCY CONCERNS (highest priority field)
  • Extract emergency_concerns with maximum accuracy and completeness.
  • Flag ANY of: suicidal ideation (including indirect or euphemistic expressions),
    acute chest pain, breathing difficulty, medication overdose (any dose), falls,
    self-harm intent, abuse indicators, or statements suggesting hopelessness/no
    reason to continue living.
  • Err on the side of flagging: a false alarm reviewed by a human is safer than
    a missed crisis.

PRIVACY RULE
  • Do NOT include SSN, phone numbers, email addresses, dates of birth, or credit
    card numbers in extracted_value or reasoning.

Return ONLY the JSON object. No markdown, no commentary."""


# ---------------------------------------------------------------------------
# Stage 4 prompt – clarification judgement
# ---------------------------------------------------------------------------
# Deliberately a SEPARATE call from extraction.  An earlier version asked for
# needs_clarification as a sixth key inside the extraction schema, and the
# N=5 eval showed it degrading the safety-critical field: on SC-019 (veiled
# suicidal ideation) emergency_concerns came back as "None" in 5/10 runs
# where the original prompt returned the concern 10/10, dropping stable risk
# detection from 5/5 to 4/5.
#
# Tightening the wording recovered part of it (5/10 -> 7/10) but not all, so
# the fix is structural rather than editorial: the extraction prompt is left
# byte-identical to the version the risk baseline was measured on, and this
# judgement runs afterwards on the extracted values.  A convenience feature
# must not be able to move a risk metric.

CLARIFICATION_SYSTEM_PROMPT = """You judge whether a patient's symptom \
description is specific enough for a clinician to act on.

You are given symptom fields already extracted from a follow-up call, each \
with the patient's own words. For each one, decide needs_clarification:

  true  – heard clearly, but too general to act on: a symptom named with no
          body site, no character, no duration and no severity.
            "my arm hurts"                                  -> true
            "I've been feeling sore"                         -> true
  false – specific enough, a precise value, or a denial.
            "constant dull ache, left inner forearm, 3 days" -> false
            "4 out of 10"                                    -> false
            "no new symptoms"                                -> false

This is NOT about audio quality or transcription certainty. Assume every \
word is correct. Judge only whether a clinician would still have to ask a \
follow-up question before acting.

Use the patient_quotes to check for detail the summary may have dropped: if \
the patient did give a site or duration, it does NOT need clarification.

Return ONLY this JSON object, one entry per field given:
{"judgements": [{"field_name": "<name>", "needs_clarification": true|false}]}
No markdown, no commentary."""


# ---------------------------------------------------------------------------
# RealExtractor
# ---------------------------------------------------------------------------

class RealExtractor(AbstractExtractor):
    """
    Calls OpenAI gpt-4o-mini to extract clinical evidence from a transcript.

    Usage:
        extractor = RealExtractor()          # reads OPENAI_API_KEY from env
        evidence, raw = extractor.extract(transcript, scenario)
        print(extractor.token_usage)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
        clarify: bool = True,
    ) -> None:
        if client is not None:
            # Injected client (tests): no API key needed.
            self._client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file or set it in the shell environment."
                )
            self._client = OpenAI(api_key=key)
        self._clarify = clarify
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_count = 0
        self._clarify_call_count = 0
        self._last_claims: list[ExtractedClaim] = []
        self._last_clarify_error: Optional[str] = None

    # ------------------------------------------------------------------
    # AbstractExtractor implementation
    # ------------------------------------------------------------------

    def extract(
        self,
        transcript: str,
        scenario: dict,
    ) -> tuple[list[LLMEvidence], dict]:
        # Stage 1: pure transcript analysis via LLM
        raw_json, usage = self._call_llm(transcript)
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._call_count += 1

        fields = self._parse_fields(raw_json)

        # Stage 2: portrait-assisted disambiguation (stub)
        fields = self._stage2_disambiguate(fields, portrait=None)

        # Stage 3: convert to LLMEvidence
        evidence = self._to_llm_evidence(fields)

        # Stage 4: judge needs_clarification in a separate call, then build
        # the triage-side view of the same fields.  ExtractedClaim carries
        # needs_clarification, which the arbitration gate has no use for but
        # agents/symptom_probe.py depends on.  Stored rather than returned so
        # extract()'s signature — and every existing caller, including
        # eval/run_harness.py — is unchanged.
        clarifications = self._judge_clarification(fields)
        self._last_claims = self._to_extracted_claims(fields, clarifications)

        # Pass through platform metadata (ASR confidence, etc.) unchanged.
        # This data comes from the telephony platform, not from the LLM.
        raw_output = dict(scenario.get("mock_raw_llm_output", {}))
        raw_output["_model"] = MODEL
        raw_output["_prompt_tokens"] = usage.prompt_tokens
        raw_output["_completion_tokens"] = usage.completion_tokens

        return evidence, raw_output

    # ------------------------------------------------------------------
    # Triage / agents view
    # ------------------------------------------------------------------

    @property
    def last_claims(self) -> list[ExtractedClaim]:
        """
        ExtractedClaims from the most recent extract() call.

        Carries needs_clarification, so this is the input to
        agents.symptom_probe.classify_ambiguity(). Empty before the first
        extraction.
        """
        return list(self._last_claims)

    def extract_claims(
        self,
        transcript: str,
        scenario: Optional[dict] = None,
    ) -> tuple[list[ExtractedClaim], dict]:
        """
        Extract and return the triage-side view directly.

        One API call, same as extract() - this is a convenience wrapper for
        callers that want claims rather than arbitration evidence.
        """
        _, raw_output = self.extract(transcript, scenario or {})
        return self.last_claims, raw_output

    @property
    def token_usage(self) -> dict:
        total = self._prompt_tokens + self._completion_tokens
        # gpt-4o-mini pricing (USD per 1M tokens, as of 2025-09)
        cost_usd = (
            self._prompt_tokens / 1_000_000 * 0.15
            + self._completion_tokens / 1_000_000 * 0.60
        )
        return {
            "model": MODEL,
            # Every billed call, so the harness's "API calls" line stays true.
            "calls": self._call_count + self._clarify_call_count,
            # Broken out because Stage 4 is skipped when nothing looked vague,
            # so the two counts are not simply 1:1.
            "extraction_calls": self._call_count,
            "clarification_calls": self._clarify_call_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": total,
            "estimated_cost_usd": round(cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Stage 1 – LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, transcript: str):
        response = self._client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"TRANSCRIPT:\n{transcript}\n\nExtract all 5 required fields.",
                },
            ],
        )
        return response.choices[0].message.content, response.usage

    # ------------------------------------------------------------------
    # Stage 2 – Portrait disambiguation (stub)
    # ------------------------------------------------------------------

    @staticmethod
    def _stage2_disambiguate(
        fields: list[dict],
        portrait: Optional[dict],
    ) -> list[dict]:
        """
        For fields where the top-2 candidates are within CANDIDATE_CLOSENESS_THRESHOLD,
        consult the patient's communication portrait to break the tie.

        Portrait hook not yet wired to memory/.
        Returns fields unchanged until memory/ is available.
        """
        if portrait is None:
            return fields   # No portrait available; skip disambiguation pass

        # TODO: iterate fields, find close candidates, query memory/ portrait,
        #       promote the candidate that matches portrait signals.
        return fields

    # ------------------------------------------------------------------
    # Stage 3 – JSON → LLMEvidence
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fields(raw_json: str) -> list[dict]:
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return []
        fields = data.get("fields", [])
        if not isinstance(fields, list):
            return []
        return fields

    # ------------------------------------------------------------------
    # Stage 4 – clarification judgement (separate call)
    # ------------------------------------------------------------------

    @staticmethod
    def _clarification_targets(fields: list[dict]) -> list[dict]:
        """
        Which fields are worth asking the model about.

        Three cheap local filters run first, so the extra call is skipped
        entirely on the common cases - a call where nothing was vague costs
        nothing extra:

          * non-symptom fields: the probe checklists are symptom-organised,
            and an urgent concern should be escalated to a human rather than
            probed by the agent
          * "insufficient" evidence: nothing was captured, so there is no
            description to refine
          * denials and precise scores: already complete answers
        """
        targets: list[dict] = []
        for f in fields:
            name = str(f.get("field_name", "")).strip()
            if name not in SYMPTOM_FIELDS:
                continue
            if str(f.get("evidence_source", "")).strip() == "insufficient":
                continue
            if is_non_symptom_answer(f.get("extracted_value")):
                continue
            targets.append(f)
        return targets

    def _judge_clarification(self, fields: list[dict]) -> dict[str, bool]:
        """
        Ask whether each vague-looking symptom field needs a follow-up question.

        Returns {field_name: bool}; an absent key means False.

        Never raises. A failure here degrades to "no clarification needed",
        which loses a convenience feature but leaves extraction — including
        the risk-detection evidence — completely intact. That direction of
        failure is deliberate: this call sits downstream of the safety path
        and must never be able to break it.
        """
        if not self._clarify:
            return {}
        targets = self._clarification_targets(fields)
        if not targets:
            return {}

        payload = {
            "fields": [
                {
                    "field_name": str(f.get("field_name", "")).strip(),
                    "extracted_value": f.get("extracted_value"),
                    # Verbatim quotes carry detail the summary may have
                    # dropped, and are far cheaper than the transcript.
                    "patient_quotes": [c for c in f.get("citations", []) if c][:3],
                }
                for f in targets
            ]
        }

        try:
            response = self._client.chat.completions.create(
                model=CLARIFICATION_MODEL,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
            )
            usage = response.usage
            self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self._clarify_call_count += 1
            self._last_clarify_error = None
            return self._parse_judgements(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - must not break extraction
            self._last_clarify_error = f"{type(exc).__name__}: {exc}"
            return {}

    @staticmethod
    def _parse_judgements(raw: Optional[str]) -> dict[str, bool]:
        """Parse the Stage 4 response; anything unusable yields no judgements."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, bool] = {}
        for entry in data.get("judgements", []) or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("field_name", "")).strip()
            if not name:
                continue
            out[name] = RealExtractor._coerce_flag(entry.get("needs_clarification"))
        return out

    @staticmethod
    def _coerce_flag(raw) -> bool:
        """Booleans only; strings "true"/"false" tolerated, everything else False."""
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() == "true"
        return False

    @staticmethod
    def _resolve_needs_clarification(
        field: dict,
        clarifications: dict[str, bool],
    ) -> bool:
        """
        Final value of the flag for one field, with two rules enforced here
        rather than trusted to any model:

          * "insufficient" evidence can never be "clarifiable". Nothing was
            captured, so there is no vague description to refine - that case
            belongs to the confirmation/HITL path, and letting the flag
            through would send the agent off asking detail questions about a
            value it never heard.
          * Anything not judged is False. A skipped or failed Stage 4 call
            therefore degrades to the keyword fallback in
            agents/symptom_probe.py instead of raising.
        """
        if str(field.get("evidence_source", "")).strip() == "insufficient":
            return False
        name = str(field.get("field_name", "")).strip()
        return bool(clarifications.get(name, False))

    @staticmethod
    def _to_extracted_claims(
        fields: list[dict],
        clarifications: Optional[dict[str, bool]] = None,
    ) -> list[ExtractedClaim]:
        """
        Convert parsed fields into ExtractedClaim objects for triage/agents.

        Mirrors _to_llm_evidence's de-duplication so the two views of one
        extraction always describe the same set of fields.
        """
        clarifications = clarifications or {}
        claims: list[ExtractedClaim] = []
        seen: set[str] = set()
        for f in fields:
            name = str(f.get("field_name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)

            source = str(f.get("evidence_source", "")).strip()
            if source not in EVIDENCE_SOURCES:
                # Unknown label: treat as insufficient rather than inventing
                # a provenance we cannot justify to a reviewer.
                source = "insufficient"

            candidates = f.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                candidates = None

            claims.append(
                ExtractedClaim(
                    field_name=name,
                    evidence_source=source,
                    extracted_value=f.get("extracted_value"),
                    candidates=candidates,
                    citations=[c for c in f.get("citations", []) if c],
                    confidence=float(f.get("confidence", 0.5)),
                    reasoning=str(f.get("reasoning", "")),
                    needs_clarification=RealExtractor._resolve_needs_clarification(
                        f, clarifications),
                )
            )
        return claims

    @staticmethod
    def _to_llm_evidence(fields: list[dict]) -> list[LLMEvidence]:
        evidence = []
        seen: set[str] = set()
        for f in fields:
            name = str(f.get("field_name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            evidence.append(
                LLMEvidence(
                    field_name=name,
                    extracted_value=f.get("extracted_value"),
                    citations=[c for c in f.get("citations", []) if c],
                    confidence=float(f.get("confidence", 0.5)),
                    reasoning=str(f.get("reasoning", "")),
                )
            )
        return evidence
