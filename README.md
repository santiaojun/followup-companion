# FollowUp Companion

**AI phone follow-up agent for chronic-disease and post-surgical patients, built on [CALL-E](https://www.heycall-e.com/).**

FollowUp Companion places real follow-up phone calls, listens for both explicit and indirect risk signals, asks the clinically useful follow-up questions a scripted IVR would skip, and routes anything it isn't confident about to a human — instead of guessing.

Built for the CALL-E hackathon. **This is a research prototype, not a medical device, and is not intended for clinical use.**

---

## Why this exists

Most automated follow-up calls collect data; they don't catch what matters. A patient who feels like they're talking to a form is less likely to mention the detail that actually needs attention — because nothing about the call invited them to.

FollowUp Companion is built around one constraint that shaped every design decision below: **safety-critical judgments are never left to a model alone.** The LLM understands language; deterministic code decides what happens next.

---

## What it does

- **Learns a communication profile from nothing.** No pre-loaded personality data, no intake form. The system starts with a neutral script and, after enough calls to trust the pattern, adapts pacing and tone based on how the patient actually talks.
- **Detects risk through two independent paths.** A deterministic keyword scanner that never depends on the model (the safety net), plus a path where the LLM can flag risk it understands semantically — indirect language like giving away possessions, or "things I never got to say" — that keyword matching alone would miss. Either path alone misses cases; together they catch more without one silently overriding the other.
- **Asks real follow-up questions.** When a patient mentions pain, the system probes for location, quality, and duration — the details a triage nurse would actually need — instead of moving straight to the next scripted item.
- **Doesn't guess when it isn't sure.** A tiered confirmation mechanism handles uncertainty differently depending on its source: mishearing (confirm by repeating back, or offer a small set of candidates), versus a vague-but-audible answer (probe for the missing detail), versus a signal that's ambiguous enough that only a human should resolve it.
- **Makes every verdict traceable.** Every decision records which dimension fired, what evidence supported it, and whether that evidence came directly from the transcript or was inferred with help from the patient's profile — so a reviewing clinician sees why the system did what it did, not just a flag.

---

## Architecture

```
providers/     CALL-E SDK integration, call requests, three-tier failover
agents/        Pre-call planning (CallPlanner), tiered confirmation gate,
               symptom-detail probing — the deterministic layers that sit
               around the one LLM call that actually needs to happen
arbitration/   Six-dimension verification gate: completeness, evidence
               traceability, out-of-distribution detection, transcription
               quality, risk signal (dual-path), PII boundary
memory/        RAG over past call summaries (Chroma) + communication
               profile store with confidence-gated personalization
eval/          Multi-sample evaluation harness — every metric below is
               measured across repeated runs, not a single lucky pass
docs/          Static project site (this repo's GitHub Pages source)
```

**Pipeline, end to end:**

```
Stage 0  Call Plan        CallPlanner reads RAG history + profile,
                           builds a personalised task + probe rules
Stage 1  Initiate          CALL-E outbound call, task + result_schema
                           injected explicitly (not left to a template)
Stage 2  Poll               Wait for CALL-E to return transcript + status
Stage 3  Extract            LLM parses free text into structured claims,
                           each tagged with its evidence source
Stage 4  Arbitrate          Six-dimension gate decides ACCEPT /
                           TRIAGE_RISK / TRIAGE_AMBIGUOUS / REJECT
Stage 5  Probe              If a symptom is under-specified, generate
                           the missing follow-up question
Stage 6  Confirm            Tiered confirmation resolves residual
                           uncertainty before the verdict is finalised
```

---

## Setup

```bash
git clone https://github.com/santiaojun/followup-companion.git
cd followup-companion
pip install -r requirements.txt   # or the equivalent for your environment
```

Create a `.env` file (never committed — see `.gitignore`):

```
CALLE_API_KEY=your_calle_key
OPENAI_API_KEY=your_openai_key
```

Run the test suite:

```bash
python -m pytest -v
```

Run the pipeline against mock data (no real call, no API cost):

```bash
python demo/run_call.py --extractor mock
```

Run it end-to-end with a real extractor and a real CALL-E call:

```bash
python demo/run_real_call.py <phone-number> --patient-id PAT-001
```

---

## What we verified, and how

Rather than asserting the safety design works, we tested whether breaking it would be caught:

**Red-team test.** We deliberately disabled the risk-detection keyword scan and re-ran the suite. Ten tests failed — not with an error, but by confidently reporting a dangerous transcript as safe (`score: 1.0`, meaning "no concern"). That failure mode — a wrong, confident answer instead of a crash — is exactly what a safety-critical system must never do silently. We restored the code and confirmed the suite returned to green.

**Multi-sample evaluation, not single-run numbers.** LLM outputs vary across identical calls. Instead of quoting a single lucky pass, we re-ran the full scenario suite five times per case and split results into three buckets: consistently correct, consistently wrong (a real bug), and unstable (model noise, not a defect — reported separately, never folded into headline metrics).

**Iteration, with the numbers that came out of it:**

| Version | Recall on indirect risk signals | False positive rate |
|---|---|---|
| Keyword-only (single path) | 37.5% | — |
| Dual-path detection (first pass) | 75.0% | elevated |
| After root-cause fixes (negation handling, traceability logic, contextual suppression) | 100% (stable, 5-sample verified) | 0% (stable, 5-sample verified) |

One known limitation was left unfixed on purpose: a confidence-scoring edge case (a negative finding sometimes reporting low confidence despite a clear quote) could be patched, but doing so measurably reduced recall on veiled suicidal ideation in testing. We chose not to trade that recall for a cleaner accuracy number, and documented the tradeoff rather than making it silently.

---

## Honest scope

This is a hackathon-built prototype, not a production system. Specifically simplified for this build:

- **Human review routing** currently marks calls for review; a full priority-tiered supervisor queue is designed but not implemented.
- **Audit storage** is file/log-based rather than a database with enforced append-only constraints.
- **CALL-E integration** has been verified with real outbound calls; call volume during development was limited by the platform's free-tier call allowance.

None of the above affects the correctness of the detection and arbitration logic itself, which is the part of this system we tested most rigorously.

---

## Safety & Ethics

This system is a decision-support tool, not a clinical decision-maker. It never diagnoses, and any indication of acute risk is routed to a human rather than resolved autonomously by the model. Deterministic rules govern every safety-critical decision path; the LLM's role is limited to language understanding and evidence proposal, never final judgment on risk. All patient names and scenarios in the demo and documentation are fictional.

---

## License

MIT. See `LICENSE`.
