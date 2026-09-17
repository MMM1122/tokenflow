"""Plan and resume paired experiments without automatically repeating paid attempts."""

import hashlib
import importlib.metadata
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path

from tokenflow.core.optimizer import Optimizer
from tokenflow.core.tokenizer import render_messages
from tokenflow.datasets import load_cases
from tokenflow.llm.providers import Provider, ProviderResult
from tokenflow.metrics.pricing import Prices, cost
from tokenflow.metrics.statistics import percentile
from tokenflow.models import BudgetExceeded, OptimizationRequest
from tokenflow.run_store import RunStore, atomic_json, atomic_text, digest, read_json, run_lock


def answer_check(answer: str, case: dict) -> float:
    text = answer.casefold()
    if any(term.casefold() in text for term in case.get("answer_forbidden", [])):
        return 0.0
    checks = case["answer_contains"]
    if not checks:
        raise ValueError("Online evaluation requires explicit answer checks")
    return sum(term.casefold() in text for term in checks) / len(checks)


def source_digest() -> str:
    root = Path(__file__).parent
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
        }
    )


def build_plan(
    dataset: Path,
    *,
    model: str | None,
    mode: str = "balanced",
    limit: int = 100,
    seed: int = 42,
    max_output_tokens: int = 512,
    prices: Prices | None = None,
    provider_type: str = "openai",
) -> dict:
    if type(limit) is not int or limit < 1:
        raise ValueError("Online case limit must be positive")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if model is not None and not model.strip():
        raise ValueError("A nonblank model is required")
    if prices and prices.model != model:
        raise ValueError("Prices model must match the requested evaluation model")
    cases = load_cases(dataset)  # Validate every row, not just the random subset.
    rng = random.Random(seed)
    rng.shuffle(cases)
    selected = cases[:limit]
    engine = Optimizer()
    planned = []
    for case in selected:
        request = OptimizationRequest.model_validate({**case["request"], "mode": mode})
        baseline = render_messages(request)
        arms = ["baseline", "optimized"]
        rng.shuffle(arms)
        item = {
            **case,
            "arms": arms,
            "left_arm": rng.choice(["baseline", "optimized"]),
            "messages": {"baseline": baseline},
            "input_tokens": {"baseline": engine.tokenizer.count(baseline)},
        }
        try:
            result = engine.optimize(request)
            item.update(
                status="ready",
                optimization_ms=result.metrics.optimization_ms,
                lossy=result.metrics.lossy,
            )
            item["messages"]["optimized"] = result.messages
            item["input_tokens"]["optimized"] = result.metrics.tokens_after
        except BudgetExceeded:
            item.update(status="budget_exceeded", optimization_ms=0.0, lossy=False)
            item["messages"]["optimized"] = ()
        planned.append(item)
    identity = {
        "format_version": 2,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "mode": mode,
        "seed": seed,
        "limit": limit,
        "requested_model": model,
        "max_output_tokens": max_output_tokens,
        "provider": provider_type,
        "encoding": engine.tokenizer.name,
        "tokenizer_version": importlib.metadata.version("tiktoken"),
        "optimizer_source_sha256": source_digest(),
        "prices": prices.model_dump() if prices else None,
    }
    return {"identity": identity, "cases": planned}


def plan_summary(plan: dict) -> dict:
    ready = [case for case in plan["cases"] if case["status"] == "ready"]
    calls = len(ready) * 2
    tokens = {
        arm: sum(case["input_tokens"][arm] for case in ready) for arm in ("baseline", "optimized")
    }
    output_cap = calls * plan["identity"]["max_output_tokens"]
    prices = plan["identity"]["prices"]
    estimate = (
        None
        if not prices
        else (
            sum(tokens.values()) * prices["input_per_million"]
            + output_cap * prices["output_per_million"]
        )
        / 1_000_000
    )
    return {
        **plan["identity"],
        "kind": "offline_evaluation_preflight",
        "cases": len(plan["cases"]),
        "ready_cases": len(ready),
        "blocked_cases": len(plan["cases"]) - len(ready),
        "planned_provider_calls": calls,
        "actual_provider_calls": 0,
        "estimated_input_tokens": tokens,
        "configured_total_output_token_cap": output_cap,
        "uncached_input_plus_output_cap_cost_estimate": estimate,
        "estimate_is_billing_cap": False,
        "answer_quality_retention": None,
        "currency": prices["currency"] if prices else None,
        "cases_by_family": {
            family: sum(c["family"] == family for c in plan["cases"])
            for family in sorted({c["family"] for c in plan["cases"]})
        },
        "case_status": [{"id": c["id"], "status": c["status"]} for c in plan["cases"]],
        "limitations": [
            "No model calls were made; this is not measured quality, cost, or latency.",
            "Input counts estimate provider framing. Cost uses uncached input and output caps.",
            "The estimate is not a hard billing limit or confirmation of model access.",
        ],
    }


def _save_calls(output: Path, ordered: list[dict]):
    atomic_text(output / "calls.jsonl", "".join(json.dumps(row) + "\n" for row in ordered))


def _summarize(manifest: dict, rows: list[dict]) -> dict:
    summary = {
        **manifest,
        "call_count": sum(row["status"] != "budget_exceeded" for row in rows),
        "call_count_semantics": "durably_started_attempts_including_uncertain",
        "evaluation_arm_records": len(rows),
        "failed_calls": sum(row["status"] != "ok" for row in rows),
        "uncertain_calls": sum(row["status"] == "uncertain" for row in rows),
        "answer_quality_retention": None,
        "product_gate": "pending_blinded_review_and_independent_dataset",
        "response_models": sorted({r["model"] for r in rows if r["status"] == "ok"}),
    }
    for arm in ("baseline", "optimized"):
        selected = [row for row in rows if row["arm"] == arm]
        good = [row for row in selected if row["status"] == "ok"]
        summary[arm] = {
            "answer_check_score": statistics.mean(r["answer_check_score"] for r in selected),
            "successful_calls": len(good),
            "input_tokens_successful_calls": sum(r["input_tokens"] for r in good),
            "cached_input_tokens_successful_calls": sum(r["cached_input_tokens"] for r in good),
            "output_tokens_successful_calls": sum(r["output_tokens"] for r in good),
            "recorded_successful_call_cost": sum(r["priced_cost"] for r in good)
            if manifest["prices"]
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
    summary["model_consistent"] = len(summary["response_models"]) == 1
    comparable = summary["failed_calls"] == 0 and summary["model_consistent"]
    base = summary["baseline"]["recorded_successful_call_cost"]
    opt = summary["optimized"]["recorded_successful_call_cost"]
    summary["cost_reduction"] = (base - opt) / base if comparable and base else None
    summary["paired_experiment_cost"] = (
        base + opt if (summary["failed_calls"] == 0 and base is not None) else None
    )
    summary["latency_change_ms"] = (
        (
            summary["optimized"]["mean_total_latency_ms_successful_calls"]
            - summary["baseline"]["mean_total_latency_ms_successful_calls"]
        )
        if comparable
        else None
    )
    return summary


def _export_reviews(output: Path, cases: list[dict], rows: dict):
    reviews, keys = [], []
    for case in cases:
        if any(rows[(case["id"], arm)]["status"] != "ok" for arm in case["arms"]):
            continue
        left = case["left_arm"]
        right = "optimized" if left == "baseline" else "baseline"
        reviews.append(
            {
                "id": case["id"],
                "family": case["family"],
                "request": case["request"],
                "left_answer": rows[(case["id"], left)]["text"],
                "right_answer": rows[(case["id"], right)]["text"],
                "left_scores": None,
                "right_scores": None,
                "left_critical_violation": None,
                "right_critical_violation": None,
            }
        )
        keys.append({"id": case["id"], "family": case["family"], "left": left, "right": right})
    for name, records in (("blind-review.jsonl", reviews), ("review-key.jsonl", keys)):
        if name == "blind-review.jsonl" and (output / name).exists():
            # Preserve reviewer edits after interrupted export/finalization.
            existing = {
                r["id"]: r for r in map(json.loads, (output / name).read_text().splitlines())
            }
            for review in reviews:
                old = existing.get(review["id"])
                if old and all(
                    old.get(k) == review[k]
                    for k in (
                        "family",
                        "request",
                        "left_answer",
                        "right_answer",
                    )
                ):
                    for key in (
                        "left_scores",
                        "right_scores",
                        "left_critical_violation",
                        "right_critical_violation",
                    ):
                        review[key] = old.get(key)
        atomic_text(output / name, "".join(json.dumps(r) + "\n" for r in records))


def run_online(
    dataset: Path,
    output: Path,
    provider: Provider,
    *,
    mode: str = "balanced",
    limit: int = 100,
    seed: int = 42,
    prices: Prices | None = None,
    resume: bool = False,
) -> dict:
    plan = build_plan(
        dataset,
        model=getattr(provider, "model", None),
        mode=mode,
        limit=limit,
        seed=seed,
        max_output_tokens=getattr(provider, "max_output_tokens", 512),
        prices=prices,
        provider_type=f"{type(provider).__module__}.{type(provider).__qualname__}",
    )
    with run_lock(output):
        if resume:
            manifest = read_json(output / "manifest.json")
            saved_plan = read_json(output / "plan.json")
            if (
                not isinstance(manifest, dict)
                or manifest.get("format_version") != 2
                or not isinstance(saved_plan, dict)
                or saved_plan.get("identity") != plan["identity"]
                or manifest.get("plan_sha256") != digest(saved_plan)
                or any(manifest.get(key) != value for key, value in plan["identity"].items())
                or manifest.get("cases") != len(saved_plan.get("cases", []))
            ):
                raise ValueError("Cannot resume: dataset, configuration, code, or plan changed")
            plan = saved_plan
        else:
            if any(path.name != ".run.lock" for path in output.iterdir()):
                raise ValueError("Use an empty output directory or pass --resume")
            manifest = {
                **plan["identity"],
                "cases": len(plan["cases"]),
                "plan_sha256": digest(plan),
                "generated_at": datetime.now(UTC).isoformat(),
            }
            atomic_json(output / "plan.json", plan)
            atomic_json(output / "manifest.json", manifest)
        store = RunStore(output)
        rows = store.validate(plan["cases"])
        for case in plan["cases"]:
            for arm in case["arms"]:
                key = (case["id"], arm)
                if key in rows:
                    continue
                row = {
                    "id": case["id"],
                    "family": case["family"],
                    "arm": arm,
                    "answer_check_score": 0.0,
                }
                if case["status"] == "budget_exceeded":
                    row["status"] = "budget_exceeded"
                elif store.path(*key, "started").exists():
                    # A model may have been charged. Never retry an ambiguous attempt.
                    row["status"] = "uncertain"
                else:
                    store.begin(*key, case["messages"][arm])
                    try:
                        response = provider.generate(tuple(case["messages"][arm]))
                        response = ProviderResult.model_validate(response)
                        row.update(
                            status="ok",
                            **response.model_dump(),
                            answer_check_score=answer_check(response.text, case),
                            priced_cost=cost(response, prices),
                            total_latency_ms=response.latency_ms
                            + (case["optimization_ms"] if arm == "optimized" else 0),
                        )
                    except Exception as exc:
                        row.update(status="provider_error", error_type=type(exc).__name__)
                store.finish(row)
                rows[key] = row
                _save_calls(output, list(rows.values()))
        ordered = [rows[(case["id"], arm)] for case in plan["cases"] for arm in case["arms"]]
        _save_calls(output, ordered)
        _export_reviews(output, plan["cases"], rows)
        summary = _summarize(manifest, ordered)
        atomic_json(output / "online.json", summary)
        return summary
