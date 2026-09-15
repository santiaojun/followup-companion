# Change Notes — 2026-09-14

Adds the pre-call / in-call agent layer (`agents/`), wires the extractor's
clarification judgement, and fixes three defects the work exposed in the
arbitration gate.

Tests: **212 → 631 passing.**

Eval (real extractor, N=5, 24 applicable scenarios):

| metric | before | after |
|---|---|---|
| risk signal recall | 8/8 = 100% | **8/8 = 100%** |
| false positive rate | 1/6 = 16.7% | **0/9 = 0%** |
| stable failures | 2 | **0** |
| unstable | 4 | **1** (annotated) |
| per-sample, 22 comparable scenarios | 92.7% | **97.3%** |
| — `false_alarm` slice | 68% | **88%** |
| cost per run | $0.0551 | $0.0562 |

Risk recall held at 40/40 per-sample in every run.

---

## 1. New package: `agents/`

The rest of the system reacts to a finished call. This package acts during
one: what to ask, and what to do when an answer is not usable.

| file | purpose |
|---|---|
| `agents/models.py` | Thresholds, enums, dataclasses |
| `agents/confirmation.py` | Tiered re-confirmation ("didn't hear it") |
| `agents/symptom_probe.py` | Detail questions ("heard it, too vague") |
| `agents/planner.py` | `CallPlanner` — the only module here that calls an LLM |
| `agents/__init__.py` | Public surface; `CallPlanner` exposed lazily so `import agents` does not pull in `openai` |

Logic is split by whether it is deterministic, following the project's
existing contract that rules never move into the LLM layer. `planner.py`
re-exports the whole surface, so `from agents.planner import ...` works for
all three tiers.

### 1a. Tiered confirmation (`confirmation.py`)

Graded by confidence, because each level costs the patient more time than
the one below it:

| confidence | level | patient hears |
|---|---|---|
| ≥ 0.85 | `NONE` | nothing — value taken as heard |
| 0.60–0.85 | `READBACK` | "You said X, is that correct?" |
| 0.40–0.60 | `CHOICE` | "Did you say stomach pain, or chest pain?" |
| < 0.40 | see below | |

Below 0.40 one attempt is spent recovering the value. If still below 0.40:
critical fields (name, drug name, dose) get one letter-by-letter `SPELL`
pass; everything else stops and routes to `triage/`.

**Termination is guaranteed twice over:** escalation is monotonic (a field
never revisits a level, so at most three utterances exist in the ladder)
*and* there is an attempt budget of 2, plus at most one extra `SPELL` pass
for critical fields. The ladder gives up on purpose — re-asking a third time
produces a worse call, not better data.

Spelling uses the NATO alphabet ("M as in Mike"): B/D/P/T/V and M/N/F/S are
exactly the pairs a narrowband phone codec destroys.

`decide_confirmation()` is a pure function; `ConfirmationTracker` adds only
per-field attempt history. A parametrised test drives 36 confidence / field
/ candidate combinations and asserts every trajectory terminates.

### 1b. Symptom detail probing (`symptom_probe.py`)

A second, independent mechanism for the opposite problem. "My arm hurts" is
transcribed perfectly at confidence 1.0 and still unusable — it has no site,
character or duration. Reading it back achieves nothing; the fix is a
clinical follow-up question.

`classify_ambiguity()` is the fork between the two mechanisms and resolves
transcription doubt **first**: probing a sentence you did not hear is
meaningless.

Checklists cover four symptom families (pain, respiratory, gastrointestinal,
wound) plus a generic fallback. Only `required` dimensions decide
incompleteness; max 3 questions per claim. Dimensions the patient already
volunteered are not re-asked, so a detailed description produces no probe.

Keyword matching is word-boundary prefix based, so `constipat` covers
constipated/constipation while `sore` does not fire on "score". Vague time
references ("recently", "lately") are deliberately excluded from the duration
keywords — those are exactly the answers that still need the question.

### 1c. Call planning (`planner.py`)

Assembles history (`memory/call_history_rag`), the communication profile
(`memory/profile_store`) and the protocol into a pre-call script.

The model writes wording; rules decide everything else:

- Pacing directives are computed from the profile in code, then passed to the
  model as constraints. The model never sees raw latency numbers.
- Personalisation is gated on `profile.use_default_script`, not model
  judgement. Below the confidence threshold the plan uses neutral wording and
  continuity topics are stripped **in code** even if the model returns them.
- Every mandatory protocol item is verified present after generation and
  restored from a template if dropped.

Degradation is total: an API failure or unparseable JSON returns a
deterministic plan with `fallback_used=True` and the reason in `notes`.
Model is `gpt-4o-mini`; `print_token_usage()` reports cost per plan
(~1,400 tokens / $0.00045).

---

## 2. `needs_clarification` wired end to end

### `triage/models.py`

`ExtractedClaim` gains `needs_clarification: bool = False` — set by the
extractor when the patient was heard clearly but said something too general
to act on. Independent of `confidence`: confidence is about the words, this
is about whether the words are specific enough to use.

### `eval/real_extractor.py`

- **New Stage 4**: a separate call that judges `needs_clarification` for the
  free-text symptom fields. Local filters skip it when nothing could need
  clarifying (non-symptom field, insufficient evidence, denial, or a precise
  score), so the extra cost only applies to calls that actually contain a
  vague symptom. Failures are swallowed and degrade to `False`.
- **Built the missing bridge**: nothing in the repo produced `ExtractedClaim`
  before. `extract()` now also normalises the same parsed fields into claims,
  exposed via `last_claims` / `extract_claims()`. `extract()`'s signature is
  unchanged, so the harness and arbitration path are untouched.
- `client=` injection and `clarify=` flag added (testability / opt-out).
- `token_usage` now breaks out `extraction_calls` vs `clarification_calls`.

Two rules are enforced in code rather than trusted to a model: `insufficient`
evidence can never be clarifiable (nothing was captured, so there is nothing
to refine), and any missing or malformed flag degrades to `False`.

### Why Stage 4 is a separate call

Asking for `needs_clarification` as a sixth key inside the extraction schema
**measurably degraded the safety-critical field**. On SC-019 (veiled suicidal
ideation) `emergency_concerns` came back as `"None"` in 5/10 runs where the
unmodified prompt returned the concern 10/10, taking stable risk detection
from 5/5 to 4/5. Mean confidence fell 0.84 → 0.74. Tightening the wording
recovered only 5/10 → 7/10.

So the fix is structural, not editorial: **`SYSTEM_PROMPT` is byte-identical
to the version the risk baseline was measured on** (asserted by a test), and
the judgement runs afterwards. A convenience feature must not be able to move
a risk metric.

---

## 3. Fixes to the arbitration gate

All three were found by the eval, not by inspection.

### 3a. `risk_signal.py` — Path 2 negation guard

The suppression check was `extracted_value.startswith("no ")`, which was
wrong in **both** directions:

- **Safety hole:** `"no reason to live"` starts with `"no "` and was
  suppressed — silencing the one path that exists to catch veiled ideation.
  (Path 1's regex never matches indirect phrasing by construction.)
- **Noise:** `"Not at all"` and `"None whatsoever."` do not start with
  `"none"` / `"no "`, so they escalated stable patients.

Replaced with an anchored whole-value lexicon. The hard part is that
`"no thoughts of ending my life"` (a denial) and `"no reason to live"`
(hopelessness) both open with "no". The discriminator is requiring an
explicit self-harm **object** after the negated noun:

| value | verdict |
|---|---|
| `no plans to hurt myself` | suppressed |
| `no plans for the future` | **fires** |
| `I have absolutely no thoughts of ending my life.` | suppressed |
| `I have thoughts of ending my life` | **fires** |
| `No, nothing urgent at all.` | suppressed |
| `No, nothing urgent, but I have no reason to live` | **fires** |

Also handles: compound answers (split on clause boundaries, suppressed only
when **every** clause denies — one concern anywhere reaches a reviewer),
trailing intensifiers ("None whatsoever."), and a bounded leading frame
(≤3 whitelisted pronoun / auxiliary / adverb tokens, never arbitrary text).

Verified 83/83 correct in both directions, built from values observed from
the live model rather than imagined. `LLM_EMERGENCY_CONFIDENCE_THRESHOLD` was
**not** touched.

### 3b. `traceability.py` — citation punctuation tolerance

When a model quotes a clause from mid-sentence it closes the quote with a
full stop the transcript does not have:

```
citation   "I have absolutely no thoughts of ending my life."
transcript "...no thoughts of ending my life and I genuinely meant it."
```

The quote is genuinely verbatim, but exact substring matching rejected it
(4/10 runs on SC-025). Trailing punctuation is now stripped before matching.
Hallucination and paraphrase still fail — the remaining text must still
appear verbatim, and internal punctuation is deliberately not normalised.

**Trap worth remembering:** the `MIN_CITATION_LENGTH` check must stay on the
citation *as given*. Measuring the stripped form silently raises the bar by
one character and rejects `"Yes I did."` (10 raw, 9 stripped) — which broke
SC-026 from 5/5 `accept` to 5/5 `triage_ambiguous`. A separate `isalnum()`
guard stops punctuation-only citations matching, since `""` is a substring of
everything.

### 3c. Both affected scenarios' notes were factually wrong

SC-021 and SC-025 both claimed the `suicidal_ideation` regex fires. It never
did: the patterns are `want to die` / `end my life` and the transcripts say
"want**ed** to die" / "end**ing** my life". Neither scenario was testing
figurative-language handling as documented. Notes replaced with measured
causes.

---

## 4. Eval harness

### `applies_to` — scenario applicability

Scenarios may declare `applies_to: ["mock"]` when their premise cannot be
reproduced by a real LLM. The harness filters on it and **prints which
scenarios were excluded**, so the denominator never shrinks silently. Absent
the key means all extractors.

`pii_leak` SC-013/014/015 are now mock-only. Their transcripts contain **no
PII at all** — the leak is injected evidence on a fabricated sixth field
(`patient_identity_verification` / `callback_number` / `contact_preferences`)
outside the five required fields. They correctly test the gate's PII rule on
the mock path (3/3); a real extractor cannot reproduce them, so `accept` is
the correct verdict there and the old "0/15" measured nothing.

### `known_unstable` — annotated instability

Scenarios may declare `known_unstable: true` + `known_unstable_reason`. The
harness labels UNSTABLE rows `[KNOWN]` vs `[NEW]`, prints the reason, counts
undiagnosed cases separately, and warns when an annotation goes **stale** (a
`known_unstable` scenario starts passing consistently).

It deliberately changes **no metric** — excluding a legitimate scenario from
scoring would hide the cost instead of recording it.

---

## 5. New scenarios: `pii_resistance` (SC-026, SC-027)

Tests what nothing previously covered: the extractor's PRIVACY RULE working
on PII that is actually present.

- **SC-026** — patient volunteers a phone number and email mid-answer.
- **SC-027** — patient reads out an SSN and explicit date of birth during
  identity verification.

Both expect `accept`: a leak into `extracted_value` or `reasoning` trips the
PII boundary and flips them to `reject`, so a failure is a genuine privacy
regression. Verified 5/5 each with zero PII in structured fields — the model
does not even carry it into citations.

SC-026's *mock* evidence deliberately places the PII-bearing quote in
`citations`, pinning `pii_boundary`'s documented "citations are NOT scanned"
decision as asserted behaviour. That decision was left alone: citations reach
clinician reviewers who already have the transcript.

SC-026 earned its place immediately by catching the guard's comma bug
(`'No, nothing urgent at all.'`) and later the `MIN_CITATION_LENGTH`
regression.

Scenario count: 25 → 27.

---

## 6. Known limitations

### 6a. `confidence` is undefined for a negative finding — OPEN

**This is a system limitation, not a scenario quirk.** It affects any call
where two or more required fields are legitimately denials; SC-021 is simply
the scenario that exposes it.

`SYSTEM_PROMPT` defines `confidence` as the model's certainty in
`extracted_value`, but never says what that means when the *answer itself* is
a negative finding ("no new symptoms", "none at all"). Is it certainty that
the answer is "none" (→ 1.0), or certainty in a value the model does not have
(→ 0.0)? The contract is silent, so the model picks — and picks
inconsistently across samples.

Measured on SC-021, N=10, both denial fields (`symptom_update` and
`emergency_concerns`):

| | `extracted_value` | `confidence` | `evidence_source` |
|---|---|---|---|
| 6/10 | `'None at all.'` | 1.0 | `direct_quote` |
| 2/10 | `'None'` | **0.00** | `direct_quote` ← self-contradictory |
| 2/10 | `'None'` | **0.00** | `insufficient` |

Both fields flip together, so ~40% of responses carry two fields below 0.30.
That is enough to fail two dimensions at once:

- `completeness` — coverage drops to 3/5 = 60%, under the 80% threshold
- `confidence` — flags `low_confidence_required_field` on both

Result: `triage_ambiguous` instead of `accept`.

Worth being clear that **the gate is behaving correctly.** An extraction that
asserts a value while reporting zero confidence in it — and labels it
`direct_quote` — is self-contradictory input, and routing that to a human is
the right response. The defect is upstream, in the extraction contract.

**Not fixed, deliberately.** A `CONFIDENCE RULES` block resolves it
completely and costs risk recall on veiled suicidal ideation:

| prompt variant | SC-021 (target) | SC-019 (veiled ideation) |
|---|---|---|
| baseline | 7/10 broken | **10/10 fires** (30 samples, 3 runs) |
| full rule (3 bullets) | **0/10** ✅ | 6/10 ❌ |
| narrowed to the denial bullet only | **0/10** ✅ | 9/10 ❌ |

Reverted both times. On `risk_indirect` scenarios the LLM path is the *only*
detector — Path 1's regex never matches indirect phrasing by construction —
so a 10–40% miss rate there is a patient-safety cost, against reviewer
minutes for the false positive. The 10/10 gate was declared before the
measurement and not relaxed after seeing the result.

Likely mechanism for the degradation: the bullets *"use low confidence only
when you could not determine the answer → set `extracted_value` to null"*
push the model into a binary (confident, or nothing), which destroys the
"uncertain but present concern" representation SC-019 depends on. The
narrowed single-bullet version still cost 1/10, so this prompt is sensitive
to edits of any size.

**If you want it fixed**, it is a one-line addition to `SYSTEM_PROMPT` — but
re-run the full N=5 eval *and* the SC-019 A/B (`--samples 5` plus 10 paired
samples on SC-019) before accepting it, and treat any drop below 10/10 as
disqualifying. An out-of-band repair call (Stage 4 style) would keep the
prompt clean, but a second call asking "how confident are you really?" would
just re-guess, so it is probably not worth building.

Annotated `known_unstable` in `test_scenarios.jsonl` rather than hidden, so
it prints as `[KNOWN]` with its reason and does not read as a new problem.
It changes no metric.

### 6b. `pii_leak` SC-013/014/015 cannot be exercised by a real extractor

Documented in §4. Mock-only by declaration; the gate's PII rule is still
fully covered on the mock path (3/3), and `pii_resistance` (§5) now covers
the real path.

### 6c. chromadb ephemeral-mode query bug — FIXED

`memory/tests/test_call_history_rag.py::TestMetadataIntegrity` flaked
intermittently (~1 run in 8). **Root cause, after capturing the traceback:**

```
chromadb.errors.InternalError: Error executing plan:
Internal error: Error finding id
  raised from chromadb/api/rust.py -> self.bindings.query(...)
```

A chromadb 1.5.9 bug: once enough `EphemeralClient` instances exist in one
process, `.query()` against a collection created *earlier* fails inside the
Rust bindings. Measured:

| scenario | failed queries |
|---|---|
| 60 ephemeral instances | **23/60** |
| 40 clients, 1 collection each | **32/40** |
| 1 client, 40 collections | 0/40 |
| 40 **persistent** stores, same sequence | **0/40** |

Not transient — an immediate retry also fails (0/20 recovered), so a retry
wrapper would have been useless.

This also explains why only *those two* tests flaked: they were the only
tests reaching `.query()` after the write fixtures had churned ~20 clients.
`retrieve_recent()` goes through `.get()`, which is unaffected, and the
unknown-patient tests short-circuit on a zero count before querying.

**Fixes:**

- `memory/tests/test_call_history_rag.py` — fixtures now use
  `tmp_path_factory` (PersistentClient) instead of in-memory mode. Stores
  are isolated by filesystem path, pytest cleans them up, and the tests now
  exercise the client production actually runs.
- `memory/call_history_rag.py` — ephemeral mode reuses one client per
  process (`_shared_ephemeral_client`), so REPL and ad-hoc use cannot hit
  the same bug. Isolation is unchanged: it comes from `collection_name`, and
  two instances sharing a name already shared a store. Covered by a new
  `TestEphemeralMode` class.
- `memory/call_history_rag.py` — the default embedding function is also
  cached per process, so the ONNX MiniLM model loads once instead of once
  per instance.
- Two tests additionally had a **hidden ranking dependency**: they asserted
  on `results[0]` with `n_results=1`, so a metadata check silently required
  CALL-A to out-rank CALL-D (whose summary also mentions medication and
  tablets) for the bare query `"blood glucose"`. They now locate the record
  by `call_id`, matching their robust sibling. Ranking keeps its dedicated
  coverage in `TestBloodGlucoseRetrieval`, which uses a discriminating query.
- `populated_rag` asserts its own document count at setup, so a population
  failure reports a count rather than surfacing as a downstream `IndexError`.

**Correction to an earlier claim in this session:** the cause was first
reported as "collections accumulating in the EphemeralClient singleton, with
a semantic query occasionally matching another test's document". That was
wrong. Direct experiment showed different-named collections are properly
isolated and a query cannot return another collection's documents; the real
failure was an exception, not a wrong result.

---

## 7. Files changed

**New**

```
agents/__init__.py                      agents/models.py
agents/confirmation.py                  agents/symptom_probe.py
agents/planner.py                       agents/tests/{__init__,test_confirmation,
                                          test_symptom_probe,test_planner}.py
eval/tests/{__init__,test_real_extractor}.py
CHANGES.md
```

**Modified**

```
triage/models.py                        needs_clarification field
eval/real_extractor.py                  Stage 4, claims bridge, client/clarify params
eval/run_harness.py                     applies_to, known_unstable, skip reporting
eval/test_scenarios.jsonl               25 -> 27; annotations; corrected notes
arbitration/dimensions/risk_signal.py   negation guard rewrite
arbitration/dimensions/traceability.py  citation punctuation tolerance
arbitration/tests/test_risk_signal.py   guard coverage, both directions
arbitration/tests/test_ambiguous.py     citation tolerance + regression test
pyproject.toml                          testpaths += agents/tests, eval/tests
```

**Test counts**

| module | tests |
|---|---|
| `arbitration/tests` | 206 |
| `agents/tests` | 257 |
| `memory/tests` | 66 |
| `eval/tests` | 59 |
| `triage/tests` | 43 |
| **total** | **631** |
