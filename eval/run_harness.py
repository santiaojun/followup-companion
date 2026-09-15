"""
FollowUp Companion – Arbitration Eval Harness

Supports single-run mode (--samples 1, default) and multi-sample mode
(--samples N, recommended N=5 for real extractor).

In multi-sample mode each scenario is run N times.  Results are classified:
  STABLE_PASS  all N runs correct             (confirmed working)
  STABLE_FAIL  all N runs same wrong answer   (confirmed system bug)
  UNSTABLE     runs disagree                  (LLM sampling noise)

Only STABLE scenarios count toward recall / FPR baselines, so systematic
bugs (SC-017, SC-020) are separated from boundary noise (SC-025).

Usage:
    python eval/run_harness.py
    python eval/run_harness.py --extractor real --samples 5
    python eval/run_harness.py --extractor mock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbitration import ArbitrationGate, ArbitrationInput, ArbitrationResult
from arbitration.models import Action, CallSignals
from eval.extractor import AbstractExtractor, MockExtractor
from eval.real_extractor import RealExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CALL_SIGNALS = CallSignals(
    avg_response_latency_ms=1100.0,
    pause_frequency=0.9,
    interruption_count=0,
    call_duration_s=180.0,
    speech_rate_wpm=115.0,
)

HUMAN_REVIEW_ACTIONS = {Action.TRIAGE_RISK, Action.TRIAGE_AMBIGUOUS}

# Stability classification labels
STABLE_PASS = "STABLE_PASS"
STABLE_FAIL = "STABLE_FAIL"
UNSTABLE    = "UNSTABLE"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SampledRun:
    """Result of a single invocation of one scenario."""
    actual_action: str
    actual_risk_flags: list[str]
    elapsed_ms: float


@dataclass
class MultiSampleResult:
    """
    Aggregated result for one scenario across N sample runs.

    stability:
      STABLE_PASS  All N runs agree AND match expected_action.
      STABLE_FAIL  All N runs agree AND do NOT match expected_action —
                   confirmed, reproducible system bug.
      UNSTABLE     Runs disagree — LLM sampling noise; excluded from baselines.
    """
    scenario_id: str
    category: str
    description: str
    expected_action: str
    expected_risk_flags: list[str]
    samples: list[SampledRun]

    # Annotation from the scenario file: this scenario's instability is
    # already diagnosed and accepted, so it should not read as a new
    # problem. Purely informational - it changes no metric, because
    # excluding a legitimate scenario from scoring would hide the cost
    # rather than record it.
    known_unstable: bool = False
    known_unstable_reason: str = ""

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def hit_count(self) -> int:
        return sum(1 for s in self.samples if s.actual_action == self.expected_action)

    @property
    def action_counts(self) -> Counter:
        return Counter(s.actual_action for s in self.samples)

    @property
    def stability(self) -> str:
        unique = {s.actual_action for s in self.samples}
        if len(unique) == 1:
            return STABLE_PASS if next(iter(unique)) == self.expected_action else STABLE_FAIL
        return UNSTABLE

    @property
    def elapsed_ms_total(self) -> float:
        return sum(s.elapsed_ms for s in self.samples)

    def outcomes_str(self) -> str:
        """e.g. 'accept(3)  triage_ambiguous(2)'"""
        return "  ".join(
            f"{action}({cnt})" for action, cnt in self.action_counts.most_common()
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class RunHarness:
    def __init__(
        self,
        scenarios_path: str | Path,
        extractor: AbstractExtractor | None = None,
        extractor_kind: str = "mock",
    ) -> None:
        self.scenarios_path = Path(scenarios_path)
        self.extractor = extractor or MockExtractor()
        self.extractor_kind = extractor_kind
        self.gate = ArbitrationGate()
        # Scenarios filtered out by their "applies_to" list, kept so the
        # report can state what was excluded instead of quietly shrinking
        # the denominator.
        self.skipped_scenarios: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, n_samples: int = 1) -> list[MultiSampleResult]:
        scenarios = self._load_scenarios()
        total = len(scenarios)
        results: list[MultiSampleResult] = []

        if n_samples > 1:
            print(
                f"  Running {total} scenarios x {n_samples} samples "
                f"= {total * n_samples} API calls",
                file=sys.stderr,
            )

        for i, sc in enumerate(scenarios, 1):
            if n_samples > 1:
                print(
                    f"  [{i:>2}/{total}] {sc['scenario_id']} ",
                    end="", file=sys.stderr, flush=True,
                )

            samples: list[SampledRun] = []
            for _ in range(n_samples):
                samples.append(self._run_one_sample(sc))
                if n_samples > 1:
                    print(".", end="", file=sys.stderr, flush=True)

            if n_samples > 1:
                unique_actions = {s.actual_action for s in samples}
                if len(unique_actions) == 1:
                    action = next(iter(unique_actions))
                    sym = "[OK]" if action == sc["expected_action"] else "[FAIL]"
                else:
                    sym = "[???]"
                print(f" {sym}", file=sys.stderr)

            results.append(MultiSampleResult(
                scenario_id=sc["scenario_id"],
                category=sc["category"],
                description=sc.get("description", ""),
                expected_action=sc["expected_action"],
                expected_risk_flags=sc.get("expected_risk_flags", []),
                samples=samples,
                known_unstable=bool(sc.get("known_unstable", False)),
                known_unstable_reason=sc.get("known_unstable_reason", ""),
            ))
        return results

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def _run_one_sample(self, scenario: dict) -> SampledRun:
        transcript = scenario["transcript"]
        t0 = time.perf_counter()

        llm_evidence, raw_llm_output = self.extractor.extract(transcript, scenario)
        arb_input = ArbitrationInput(
            call_id=scenario["scenario_id"],
            patient_id=f"PATIENT-{scenario['scenario_id']}",
            transcript=transcript,
            llm_evidence=llm_evidence,
            call_signals=DEFAULT_CALL_SIGNALS,
            raw_llm_output=raw_llm_output,
        )
        arb_result: ArbitrationResult = self.gate.evaluate(arb_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return SampledRun(
            actual_action=arb_result.action.value,
            actual_risk_flags=arb_result.risk_flags,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # JSONL loader
    # ------------------------------------------------------------------

    def _load_scenarios(self) -> list[dict]:
        """
        Load scenarios applicable to the current extractor.

        A scenario may declare `applies_to: ["mock"]` when its premise cannot
        be reproduced by a real LLM. SC-013/014/015 are the case in point:
        their transcripts contain no PII at all, and the leak lives in
        injected evidence on a fabricated sixth field, so they test the
        gate's PII rule rather than any model behaviour. Running them
        against a real extractor scores a guaranteed "failure" that measures
        nothing. Absent the key, a scenario applies to every extractor.
        """
        scenarios = []
        self.skipped_scenarios = []
        with open(self.scenarios_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sc = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  WARNING: skipping malformed line {lineno}: {exc}")
                    continue

                applies_to = sc.get("applies_to")
                if applies_to and self.extractor_kind not in applies_to:
                    self.skipped_scenarios.append(sc)
                    continue
                scenarios.append(sc)
        return scenarios


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[MultiSampleResult]) -> dict:
    """
    Stability-aware metrics.

    For recall and FPR, UNSTABLE scenarios are excluded from both numerator
    and denominator.  Only STABLE_PASS / STABLE_FAIL scenarios count, so
    LLM sampling noise does not distort the baselines.
    """
    total = len(results)
    n_sp = sum(1 for r in results if r.stability == STABLE_PASS)
    n_sf = sum(1 for r in results if r.stability == STABLE_FAIL)
    n_un = sum(1 for r in results if r.stability == UNSTABLE)
    n_samples = results[0].n if results else 1

    # Category breakdown
    categories: dict[str, list[MultiSampleResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    cat_metrics: dict = {}
    for cat, cat_results in categories.items():
        c_sp = sum(1 for r in cat_results if r.stability == STABLE_PASS)
        c_sf = sum(1 for r in cat_results if r.stability == STABLE_FAIL)
        c_un = sum(1 for r in cat_results if r.stability == UNSTABLE)
        stable_n = c_sp + c_sf
        cat_metrics[cat] = {
            "total":       len(cat_results),
            "stable_pass": c_sp,
            "stable_fail": c_sf,
            "unstable":    c_un,
            "accuracy":    c_sp / stable_n if stable_n else None,
            "results":     cat_results,
        }

    # Risk recall – stable scenarios only
    all_risk = [r for r in results if r.expected_action == Action.TRIAGE_RISK.value]
    stable_risk = [r for r in all_risk if r.stability != UNSTABLE]
    risk_recall_num = sum(1 for r in stable_risk if r.stability == STABLE_PASS)
    risk_recall_den = len(stable_risk)
    risk_unstable_n = len(all_risk) - len(stable_risk)

    # FPR – stable scenarios only
    all_accept = [r for r in results if r.expected_action == Action.ACCEPT.value]
    stable_accept = [r for r in all_accept if r.stability != UNSTABLE]
    fp_num = sum(1 for r in stable_accept if r.stability == STABLE_FAIL)
    fp_den = len(stable_accept)
    fp_unstable_n = len(all_accept) - len(stable_accept)

    timings_ms = [s.elapsed_ms for r in results for s in r.samples]

    return {
        "total":        total,
        "n_samples":    n_samples,
        "stable_pass":  n_sp,
        "stable_fail":  n_sf,
        "unstable":     n_un,
        "category":     cat_metrics,
        "risk_recall":  {
            "num":           risk_recall_num,
            "den":           risk_recall_den,
            "unstable_count": risk_unstable_n,
        },
        "false_positive": {
            "num":           fp_num,
            "den":           fp_den,
            "unstable_count": fp_unstable_n,
        },
        "timing_ms": {
            "total": sum(timings_ms),
            "avg":   sum(timings_ms) / len(timings_ms) if timings_ms else 0.0,
            "max":   max(timings_ms) if timings_ms else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    results: list[MultiSampleResult],
    metrics: dict,
    skipped: Sequence[dict] = (),
    extractor_kind: str = "",
) -> None:
    W = 76
    n = metrics["n_samples"]
    multi = n > 1

    # ── Header ────────────────────────────────────────────────────────
    print("=" * W)
    if multi:
        print(f"  FOLLOWUP COMPANION - ARBITRATION EVAL HARNESS  (N={n} samples/scenario)")
    else:
        print("  FOLLOWUP COMPANION - ARBITRATION EVAL HARNESS")
    print("=" * W)

    t = metrics["timing_ms"]
    if multi:
        total_calls = metrics["total"] * n
        print(
            f"  Scenarios : {metrics['total']}  |  Samples each : {n}  |  "
            f"Total API calls : {total_calls}"
        )
        print(
            f"  Stable-Pass : {metrics['stable_pass']}  |  "
            f"Stable-Fail : {metrics['stable_fail']}  |  "
            f"Unstable : {metrics['unstable']}"
        )
    else:
        print(
            f"  Scenarios : {metrics['total']}  |  "
            f"Passed : {metrics['stable_pass']}  |  "
            f"Failed : {metrics['stable_fail']}"
        )
    print(
        f"  Total time: {t['total']:.1f}ms  |  "
        f"Avg/call: {t['avg']:.2f}ms  |  "
        f"Max: {t['max']:.2f}ms"
    )
    if skipped:
        ids = ", ".join(sc.get("scenario_id", "?") for sc in skipped)
        label = extractor_kind or "this extractor"
        print(
            f"  Not applicable to {label} ({len(skipped)} excluded): {ids}"
        )
        print(
            f"    (declared applies_to; see the scenario notes for why)"
        )
    print()

    # ── Overall accuracy line (single-run compat) ─────────────────────
    if not multi:
        passed = metrics["stable_pass"]
        total  = metrics["total"]
        acc    = passed / total if total else 0.0
        bar    = "OK" if acc == 1.0 else ("~~" if acc >= 0.8 else "!!")
        print(f"  OVERALL ACCURACY  {bar}  {passed}/{total}  ({acc:.1%})")
        print()

    # ── Exclusion notes (multi only) ──────────────────────────────────
    if multi:
        rr = metrics["risk_recall"]
        fp = metrics["false_positive"]
        if rr["unstable_count"]:
            print(
                f"  NOTE: {rr['unstable_count']} risk scenario(s) classified UNSTABLE "
                f"-> excluded from recall denominator"
            )
        if fp["unstable_count"]:
            print(
                f"  NOTE: {fp['unstable_count']} accept scenario(s) classified UNSTABLE "
                f"-> excluded from FPR denominator"
            )
        if rr["unstable_count"] or fp["unstable_count"]:
            print()

    # ── Category breakdown ─────────────────────────────────────────────
    if multi:
        print(f"  {'CATEGORY':<22} {'N':>4}  {'S-PASS':>6}  {'S-FAIL':>6}  {'UNSTBL':>6}  {'ACCUR*':>7}")
        print(f"  {'-'*22}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
        for cat, cm in sorted(metrics["category"].items()):
            acc = cm["accuracy"]
            if acc is None:
                bar, acc_str = "~~", "   n/a "
            else:
                bar     = "OK" if acc == 1.0 else ("~~" if acc >= 0.67 else "!!")
                acc_str = f"{acc:7.1%}"
            print(
                f"  {bar} {cat:<20}  {cm['total']:>4}  "
                f"{cm['stable_pass']:>6}  {cm['stable_fail']:>6}  "
                f"{cm['unstable']:>6}  {acc_str}"
            )
        print(f"  * accuracy = S-PASS / (S-PASS + S-FAIL) -- UNSTABLE excluded")
    else:
        print(f"  {'CATEGORY':<22} {'N':>4}  {'CORRECT':>7}  {'ACCURACY':>9}")
        print(f"  {'-'*22}  {'-'*4}  {'-'*7}  {'-'*9}")
        for cat, cm in sorted(metrics["category"].items()):
            acc = cm["accuracy"]
            bar = "OK" if acc == 1.0 else ("~~" if acc is not None and acc >= 0.67 else "!!")
            suffix = ""
            if cat in ("risk_signal", "risk_indirect"):
                rr2 = metrics["risk_recall"]
                suffix = f"  recall {rr2['num']}/{rr2['den']} overall"
            elif cat in ("normal", "false_alarm"):
                fp2 = metrics["false_positive"]
                suffix = f"  FPR {fp2['num']}/{fp2['den']} overall"
            acc_val = acc if acc is not None else 0.0
            print(
                f"  {bar} {cat:<20}  {cm['total']:>4}  "
                f"{cm['stable_pass']:>4}/{cm['total']:<2}  "
                f"  {acc_val:>7.1%}{suffix}"
            )
    print()

    # ── Key metrics ───────────────────────────────────────────────────
    stable_note = "  [stable scenarios only -- UNSTABLE excluded]" if multi else ""
    rr     = metrics["risk_recall"]
    rr_pct = rr["num"] / rr["den"] if rr["den"] else 0.0
    rr_flag = "OK  CRITICAL OK" if rr_pct == 1.0 else "!!  CRITICAL FAILURE"
    print(f"  RISK SIGNAL RECALL  : {rr['num']}/{rr['den']} = {rr_pct:.1%}  {rr_flag}{stable_note}")

    fp     = metrics["false_positive"]
    fp_pct = fp["num"] / fp["den"] if fp["den"] else 0.0
    fp_flag = "OK" if fp_pct == 0.0 else ("~~" if fp_pct <= 0.33 else "!!")
    print(
        f"  FALSE POSITIVE RATE : {fp['num']}/{fp['den']} = {fp_pct:.1%}  {fp_flag}"
        f"  (expected accept -> human review){stable_note}"
    )
    print()

    # ── Stable-fail and unstable sections (multi only) ────────────────
    if multi:
        stable_fails = [r for r in results if r.stability == STABLE_FAIL]
        unstables    = [r for r in results if r.stability == UNSTABLE]

        print("  STABLE FAILURES  (confirmed bugs: same wrong answer across all samples)")
        if stable_fails:
            for r in stable_fails:
                consensus = r.action_counts.most_common(1)[0][0]
                print(
                    f"  !! {r.scenario_id:<10} [{r.category:<18}]  "
                    f"expected {r.expected_action:<20} "
                    f"got {consensus}({r.n}/{r.n})"
                )
        else:
            print("  (none)")
        print()

        new_unstables = [r for r in unstables if not r.known_unstable]
        known_unstables = [r for r in unstables if r.known_unstable]

        print("  UNSTABLE CASES  (LLM noise: excluded from recall / FPR baselines)")
        if unstables:
            for r in unstables:
                tag = "[KNOWN]" if r.known_unstable else "[NEW]  "
                print(
                    f"  ~~ {tag} {r.scenario_id:<10} [{r.category:<18}]  "
                    f"expected {r.expected_action:<20} "
                    f"hit {r.hit_count}/{r.n}  |  {r.outcomes_str()}"
                )
            if known_unstables:
                print()
                print("    KNOWN-UNSTABLE (diagnosed and accepted):")
                for r in known_unstables:
                    print(f"      {r.scenario_id}: {r.known_unstable_reason}")
            if new_unstables:
                print()
                print(
                    f"    {len(new_unstables)} UNDIAGNOSED unstable scenario(s): "
                    f"{', '.join(r.scenario_id for r in new_unstables)}"
                )
        else:
            print("  (none)")

        # A scenario annotated known-unstable that now passes consistently
        # means the annotation is stale and should be removed, so say so
        # rather than letting it sit there forever.
        stale = [r for r in results if r.known_unstable and r.stability == STABLE_PASS]
        if stale:
            print()
            for r in stale:
                print(
                    f"  NOTE: {r.scenario_id} is annotated known_unstable but passed "
                    f"{r.hit_count}/{r.n} - annotation is stale, consider removing it."
                )
        print()

    # ── Per-scenario detail ────────────────────────────────────────────
    if multi:
        print(
            f"  {'ID':<10} {'CATEGORY':<18} {'EXPECTED':<22}"
            f"{'STAB':>7}  {'HIT':>5}  OUTCOMES"
        )
        print(
            f"  {'-'*10} {'-'*18} {'-'*22}"
            f"{'-'*7}  {'-'*5}  {'-'*30}"
        )
        for r in results:
            stab_short = {
                STABLE_PASS: "S-PASS",
                STABLE_FAIL: "S-FAIL",
                UNSTABLE:    "UNSTBL",
            }[r.stability]
            flag = {STABLE_PASS: "OK", STABLE_FAIL: "!!", UNSTABLE: "~~"}[r.stability]
            print(
                f"  {flag} {r.scenario_id:<10} {r.category:<18} "
                f"{r.expected_action:<22}"
                f"{stab_short:>7}  {r.hit_count:>2}/{r.n:<2}  {r.outcomes_str()}"
            )
    else:
        print(
            f"  {'ID':<10} {'CATEGORY':<18} {'EXPECTED':<18} "
            f"{'ACTUAL':<18} {'ms':>6}  RESULT"
        )
        print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*18} {'-'*6}  {'-'*6}")
        for r in results:
            s    = r.samples[0]
            flag = "PASS" if r.stability == STABLE_PASS else "FAIL"
            detail = (
                f"  (expected {r.expected_action!r} but got {s.actual_action!r})"
                if r.stability != STABLE_PASS else ""
            )
            print(
                f"  {r.scenario_id:<10} {r.category:<18} {r.expected_action:<18} "
                f"{s.actual_action:<18} {s.elapsed_ms:>5.1f}ms  {flag}{detail}"
            )

    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FollowUp Companion eval harness.",
    )
    parser.add_argument(
        "--scenarios",
        default=str(Path(__file__).parent / "test_scenarios.jsonl"),
        help="Path to the JSONL scenario file.",
    )
    parser.add_argument(
        "--extractor",
        choices=["mock", "real"],
        default="mock",
        help="Which extractor to use (default: mock).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Samples per scenario (default: 1).  "
            "Use --samples 5 with --extractor real to separate "
            "stable system bugs from LLM sampling noise."
        ),
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")

    extractor: AbstractExtractor
    if args.extractor == "mock":
        extractor = MockExtractor()
    else:
        extractor = RealExtractor()
        if args.samples == 1:
            print(
                "  TIP: run with --samples 5 to separate stable bugs from LLM noise.",
                file=sys.stderr,
            )

    harness = RunHarness(
        scenarios_path=args.scenarios,
        extractor=extractor,
        extractor_kind=args.extractor,
    )
    results = harness.run(n_samples=args.samples)
    metrics = compute_metrics(results)
    print_report(
        results, metrics,
        skipped=harness.skipped_scenarios,
        extractor_kind=args.extractor,
    )

    # Token usage (real extractor only)
    usage = extractor.token_usage
    if usage:
        print("  TOKEN USAGE SUMMARY")
        print(f"  {'Model':<22}: {usage['model']}")
        print(f"  {'API calls':<22}: {usage['calls']}")
        print(f"  {'Prompt tokens':<22}: {usage['prompt_tokens']:,}")
        print(f"  {'Completion tokens':<22}: {usage['completion_tokens']:,}")
        print(f"  {'Total tokens':<22}: {usage['total_tokens']:,}")
        print(f"  {'Estimated cost':<22}: ${usage['estimated_cost_usd']:.4f} USD")
        print("=" * 76)

    # Exit non-zero if risk recall (stable-only) < 100%
    rr = metrics["risk_recall"]
    if rr["den"] > 0 and rr["num"] < rr["den"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
