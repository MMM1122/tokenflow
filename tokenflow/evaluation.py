"""Offline and paired-provider evaluation; never confuse proxies with answer quality."""

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import Field

from tokenflow.core.optimizer import Optimizer
from tokenflow.core.tokenizer import render_messages
from tokenflow.llm.providers import Provider, ProviderResult
from tokenflow.models import BudgetExceeded, OptimizationRequest, StrictModel


class Prices(StrictModel):
    """Explicit per-million-token prices; never infer current prices from model names."""

    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    cached_input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(min_length=1)
    as_of: str = Field(min_length=1)
    model: str = Field(min_length=1)


def cost(result: ProviderResult, prices: Prices | None) -> float | None:
    if prices is None:
        return None
    if result.cached_input_tokens > result.input_tokens:
        raise ValueError("Provider cached tokens cannot exceed input tokens")
    return (
        (result.input_tokens - result.cached_input_tokens) * prices.input_per_million
        + result.cached_input_tokens * prices.cached_input_per_million
        + result.output_tokens * prices.output_per_million
    ) / 1_000_000


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("Dataset must be nonempty with unique case ids")
    for case in cases:
        OptimizationRequest.model_validate(case["request"])
        for key in ("family", "category", "split", "required_context", "answer_contains"):
            if key not in case:
                raise ValueError(f"Missing fixture field: {key}")
    return cases


def _aggregate(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["status"] == "ok"]
    rates = [row.get("reduction_rate", 0) for row in rows]
    total_before = sum(row["tokens_before"] for row in rows)
    total_after = sum(row.get("tokens_after", row["tokens_before"]) for row in rows)
    return {
        "cases": len(rows),
        "failures": len(rows) - len(successes),
        "mean_reduction_rate": statistics.mean(rates),
        "median_reduction_rate": statistics.median(rates),
        "weighted_reduction_rate": (total_before - total_after) / total_before,
        "tokens_before": total_before,
        "tokens_after_or_failed_baseline": total_after,
        "all_required_context_retained_rate": sum(
            row.get("required_context_retained", False) for row in rows
        )
        / len(rows),
        "all_protected_content_retained_rate": sum(
            row.get("protected_content_retained", False) for row in rows
        )
        / len(rows),
        "overhead_p50_ms": percentile([row["optimization_ms"] for row in successes], 0.5),
        "overhead_p95_ms": percentile([row["optimization_ms"] for row in successes], 0.95),
        "lossy_cases": sum(row.get("lossy", False) for row in rows),
    }


def run_offline(dataset: Path, repeats: int = 3, encoding: str = "o200k_base") -> dict:
    from tokenflow.core.classifier import protection_reason
    from tokenflow.core.tokenizer import Tokenizer

    if repeats < 1:
        raise ValueError("repeats must be positive")
    cases = load_cases(dataset)
    started = perf_counter()
    engine = Optimizer(Tokenizer(encoding))
    init_ms = (perf_counter() - started) * 1000
    engine.tokenizer.count_text("warmup")
    rows = []
    for mode in ("conservative", "balanced"):
        for case in cases:
            request = OptimizationRequest.model_validate({**case["request"], "mode": mode})
            before = engine.tokenizer.count(render_messages(request))
            row = {
                "id": case["id"],
                "family": case["family"],
                "category": case["category"],
                "split": case["split"],
                "mode": mode,
                "tokens_before": before,
                "length": "short" if before < 512 else "medium" if before < 2000 else "long",
            }
            try:
                results = [engine.optimize(request) for _ in range(repeats)]
                result = results[-1]
                # Evaluate original unescaped content, not JSON's escaped representation.
                retained = "\n".join(
                    [
                        request.instructions,
                        request.query,
                        *(m.content for m in request.history),
                        *(d.content for d in result.documents),
                    ]
                )
                protected_docs = {
                    d.id: d.content for d in request.documents if protection_reason(d, request)
                }
                kept_docs = {d.id: d.content for d in result.documents}
                history_start = 2 if request.instructions else 1
                history_ok = tuple(result.messages[history_start:-1]) == tuple(
                    {"role": m.role, "content": m.content} for m in request.history
                )
                query_ok = json.loads(result.messages[-1]["content"])["query"] == request.query
                instructions_ok = not request.instructions or (
                    result.messages[1] == {"role": "developer", "content": request.instructions}
                )
                row.update(status="ok", **result.metrics.model_dump())
                row["optimization_ms"] = statistics.median(
                    r.metrics.optimization_ms for r in results
                )
                row["required_context_retained"] = all(
                    fact in retained for fact in case["required_context"]
                )
                row["protected_content_retained"] = (
                    all(kept_docs.get(key) == value for key, value in protected_docs.items())
                    and history_ok
                    and query_ok
                    and instructions_ok
                )
                row["actions"] = dict(Counter(d.action for d in result.decisions))
            except BudgetExceeded:
                row.update(status="budget_exceeded")
            rows.append(row)
    summaries = {
        mode: _aggregate([r for r in rows if r["mode"] == mode])
        for mode in ("conservative", "balanced")
    }
    groups = defaultdict(list)
    for row in rows:
        for dimension in ("category", "split", "length"):
            groups[f"{row['mode']}/{dimension}/{row[dimension]}"].append(row)
    balanced = summaries["balanced"]
    functional = all(
        s["failures"] == 0
        and s["all_protected_content_retained_rate"] == 1
        and s["all_required_context_retained_rate"] == 1
        for s in summaries.values()
    )
    return {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "generated_at": datetime.now(UTC).isoformat(),
        "kind": "synthetic_offline_regression",
        "cases": len(cases),
        "independent_scenario_families": len({c["family"] for c in cases}),
        "repeats": repeats,
        "encoding": encoding,
        "tokenizer_initialization_ms": init_ms,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("tiktoken", "pydantic")},
        },
        "summary": summaries,
        "groups": {key: _aggregate(value) for key, value in groups.items()},
        "functional_fixture_gate": functional,
        "engineering_target_met": bool(
            functional
            and balanced["mean_reduction_rate"] >= 0.2
            and balanced["overhead_p95_ms"] <= 100
        ),
        "product_gate": "not_evaluated_requires_real_calls_and_independent_quality_review",
        "answer_quality_retention": None,
        "cost_reduction": None,
        "llm_latency_change_ms": None,
        "limitations": [
            "Synthetic related variants are not independent real conversations.",
            "Required-context retention is not answer-quality retention.",
            "Serialized BPE counts differ from provider billing tokens.",
            "High duplicate density favors deduplication; inspect control cases.",
        ],
        "rows": rows,
    }


def write_offline_report(report: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "offline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# TokenFlow offline regression results",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"{report['cases']} synthetic cases / {report['independent_scenario_families']} "
        "scenario families. No live LLM calls. Each case is measured three times by default.",
        "",
        "Token counts are local serialized-message BPE estimates, not billing usage.",
        "",
    ]
    for mode, summary in report["summary"].items():
        lines += [
            f"## {mode}",
            "",
            f"- Mean input reduction: {summary['mean_reduction_rate']:.2%}",
            f"- Median input reduction: {summary['median_reduction_rate']:.2%}",
            f"- Weighted input reduction: {summary['weighted_reduction_rate']:.2%}",
            f"- Warm overhead p95: {summary['overhead_p95_ms']} ms",
            f"- Required-context retention: {summary['all_required_context_retained_rate']:.2%}",
            f"- Protected-content retention: {summary['all_protected_content_retained_rate']:.2%}",
            f"- Failures: {summary['failures']} / {summary['cases']}",
            "",
        ]
    lines += [
        "## Decision",
        "",
        f"Engineering target met: {report['engineering_target_met']}.",
        "",
        "Product gate: **not evaluated**. Answer quality, billed cost savings and model "
        "latency are unmeasured. Do not claim the product maintains baseline answer quality.",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "Per-case values, category/length/split breakdowns, dataset hash and environment "
        "versions are in offline.json. See docs/benchmark-protocol.md for acceptance rules.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def answer_check(answer: str, case: dict) -> float:
    text = answer.casefold()
    if any(term.casefold() in text for term in case.get("answer_forbidden", [])):
        return 0.0
    checks = case["answer_contains"]
    if not checks:
        raise ValueError("Online evaluation requires explicit answer checks")
    return sum(term.casefold() in text for term in checks) / len(checks)


def run_online(
    dataset: Path,
    output: Path,
    provider: Provider,
    *,
    mode: str = "balanced",
    limit: int = 100,
    seed: int = 42,
    prices: Prices | None = None,
) -> dict:
    """Explicitly paid path. Save each completed call before starting the next."""
    if limit < 1:
        raise ValueError("Online case limit must be positive")
    cases = load_cases(dataset)
    rng = random.Random(seed)
    rng.shuffle(cases)
    cases = cases[:limit]
    if not cases:
        raise ValueError("Online case limit must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous paid-run evidence")
    output.mkdir(parents=True, exist_ok=True)
    engine = Optimizer()
    rows, reviews, mapping = [], [], []
    provider_calls = 0
    manifest = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "mode": mode,
        "seed": seed,
        "provider": type(provider).__name__,
        "requested_model": getattr(provider, "model", None),
        "max_output_tokens": getattr(provider, "max_output_tokens", None),
        "encoding": engine.tokenizer.name,
        "cases": len(cases),
        "generated_at": datetime.now(UTC).isoformat(),
        "prices": prices.model_dump() if prices else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (output / "calls.jsonl").open("w") as calls:
        for case in cases:
            request = OptimizationRequest.model_validate({**case["request"], "mode": mode})
            try:
                optimized = engine.optimize(request)
            except BudgetExceeded:
                for arm in ("baseline", "optimized"):
                    row = {
                        "id": case["id"],
                        "family": case["family"],
                        "arm": arm,
                        "status": "budget_exceeded",
                        "answer_check_score": 0.0,
                    }
                    rows.append(row)
                    calls.write(json.dumps(row) + "\n")
                calls.flush()
                continue
            arms = ["baseline", "optimized"]
            rng.shuffle(arms)
            pair = {}
            for arm in arms:
                row = {"id": case["id"], "family": case["family"], "arm": arm}
                messages = render_messages(request) if arm == "baseline" else optimized.messages
                try:
                    provider_calls += 1
                    response = provider.generate(messages)
                    row.update(
                        status="ok",
                        **response.model_dump(),
                        answer_check_score=answer_check(response.text, case),
                        priced_cost=cost(response, prices),
                        total_latency_ms=response.latency_ms
                        + (optimized.metrics.optimization_ms if arm == "optimized" else 0),
                    )
                    pair[arm] = response.text
                except Exception as exc:
                    row.update(
                        status="provider_error",
                        error_type=type(exc).__name__,
                        answer_check_score=0.0,
                    )
                rows.append(row)
                calls.write(json.dumps(row, ensure_ascii=False) + "\n")
                calls.flush()
            if len(pair) == 2:
                left = rng.choice(["baseline", "optimized"])
                right = "optimized" if left == "baseline" else "baseline"
                reviews.append(
                    {
                        "id": case["id"],
                        "family": case["family"],
                        "request": case["request"],
                        "left_answer": pair[left],
                        "right_answer": pair[right],
                        "left_scores": None,
                        "right_scores": None,
                        "left_critical_violation": None,
                        "right_critical_violation": None,
                    }
                )
                mapping.append({"id": case["id"], "left": left, "right": right})
    summary = {
        **manifest,
        "call_count": provider_calls,
        "evaluation_arm_records": len(rows),
        "failed_calls": sum(r["status"] != "ok" for r in rows),
        "answer_quality_retention": None,
        "product_gate": "pending_blinded_review_and_independent_dataset",
    }
    for arm in ("baseline", "optimized"):
        selected = [r for r in rows if r["arm"] == arm]
        good = [r for r in selected if r["status"] == "ok"]
        summary[arm] = {
            "answer_check_score": statistics.mean(r["answer_check_score"] for r in selected),
            "successful_calls": len(good),
            "input_tokens_successful_calls": sum(r["input_tokens"] for r in good),
            "output_tokens_successful_calls": sum(r["output_tokens"] for r in good),
            "recorded_successful_call_cost": sum(r["priced_cost"] for r in good)
            if prices
            else None,
            "mean_total_latency_ms_successful_calls": statistics.mean(
                r["total_latency_ms"] for r in good
            )
            if good
            else None,
            "p95_total_latency_ms_successful_calls": percentile(
                [r["total_latency_ms"] for r in good], 0.95
            ),
        }
    all_ok = summary["failed_calls"] == 0
    base_cost = summary["baseline"]["recorded_successful_call_cost"]
    opt_cost = summary["optimized"]["recorded_successful_call_cost"]
    summary["cost_reduction"] = (
        (base_cost - opt_cost) / base_cost
        if (all_ok and base_cost is not None and base_cost > 0)
        else None
    )
    summary["paired_experiment_cost"] = base_cost + opt_cost if all_ok and prices else None
    summary["latency_change_ms"] = (
        (
            summary["optimized"]["mean_total_latency_ms_successful_calls"]
            - summary["baseline"]["mean_total_latency_ms_successful_calls"]
        )
        if all_ok
        else None
    )
    for name, records in (("blind-review.jsonl", reviews), ("review-key.jsonl", mapping)):
        (output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
    (output / "online.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_reviews(run: Path) -> dict:
    """Consume completed blinded rubrics; leave generalization gate pending."""
    dimensions = {"correctness", "completeness", "instruction_adherence", "grounding"}
    summary = json.loads((run / "online.json").read_text())
    reviews = [json.loads(line) for line in (run / "blind-review.jsonl").read_text().splitlines()]
    key_rows = [json.loads(line) for line in (run / "review-key.jsonl").read_text().splitlines()]
    keys = {row["id"]: row for row in key_rows}
    if (
        len(reviews) != summary["cases"]
        or len(keys) != summary["cases"]
        or len({r["id"] for r in reviews}) != len(reviews)
        or summary["failed_calls"]
    ):
        raise ValueError("All pairs must succeed and receive one complete review")
    families = defaultdict(list)
    baseline_by_family, optimized_by_family = defaultdict(list), defaultdict(list)
    critical = 0
    for review in reviews:
        scores = {}
        for side in ("left", "right"):
            rubric = review[f"{side}_scores"]
            violation = review[f"{side}_critical_violation"]
            if (
                not isinstance(rubric, dict)
                or set(rubric) != dimensions
                or any(type(v) is not int or not 0 <= v <= 4 for v in rubric.values())
                or type(violation) is not bool
            ):
                raise ValueError("Each side needs four integer 0–4 scores and a boolean violation")
            arm = keys[review["id"]][side]
            scores[arm] = statistics.mean(rubric.values()) / 4
            if arm == "optimized" and violation:
                critical += 1
        baseline_by_family[review["family"]].append(scores["baseline"])
        optimized_by_family[review["family"]].append(scores["optimized"])
        families[review["family"]].append(scores["optimized"] - scores["baseline"])
    # Equal family weights avoid treating correlated context variants as independent.
    differences = [statistics.mean(values) for values in families.values()]
    if len(differences) < 2:
        raise ValueError("Need at least two independent families for a confidence interval")
    rng = random.Random(42)
    boot = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000)]
    base_mean = statistics.mean(statistics.mean(v) for v in baseline_by_family.values())
    opt_mean = statistics.mean(statistics.mean(v) for v in optimized_by_family.values())
    retention = opt_mean / base_mean if base_mean else None
    interval = [percentile(boot, 0.025), percentile(boot, 0.975)]
    return {
        "reviewed_pairs": len(reviews),
        "families": len(differences),
        "quality_weighting": "equal_family",
        "baseline_quality": base_mean,
        "optimized_quality": opt_mean,
        "answer_quality_retention": retention,
        "paired_difference_ci95": interval,
        "critical_optimized_violations": critical,
        "sample_quality_gate_passed": bool(
            base_mean >= 0.8 and retention >= 0.95 and interval[0] >= -0.05 and critical == 0
        ),
        "product_gate": "requires_independent_dataset_provenance_and_cost_latency_review",
    }
