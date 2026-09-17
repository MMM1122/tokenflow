import argparse
import json
import os
from pathlib import Path

from tokenflow.core.optimizer import Optimizer
from tokenflow.dataset_audit import audit_dataset
from tokenflow.evaluation import (
    Prices,
    evaluate_reviews,
    run_offline,
    run_online,
    write_offline_report,
)
from tokenflow.experiments import build_plan, plan_summary
from tokenflow.models import BudgetExceeded, OptimizationRequest
from tokenflow.run_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description="TokenFlow context optimizer and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize", help="Optimize one JSON request without any LLM call")
    optimize.add_argument("request", type=Path)
    audit = sub.add_parser("audit-dataset", help="Inspect benchmark isolation without model calls")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)
    benchmark = sub.add_parser("benchmark", help="Run offline synthetic evaluation")
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--repeats", type=int, default=3)
    online = sub.add_parser("evaluate-online", help="Run paid paired LLM evaluation")
    online.add_argument("dataset", type=Path)
    online.add_argument("--output", type=Path, required=True)
    online.add_argument("--model", required=True)
    online.add_argument("--limit", type=int, default=100)
    online.add_argument("--mode", choices=["conservative", "balanced"], default="balanced")
    online.add_argument("--prices", type=Path)
    online.add_argument("--seed", type=int, default=42)
    online.add_argument("--max-output-tokens", type=int, default=512)
    online.add_argument("--resume", action="store_true")
    plan = sub.add_parser("plan-evaluation", help="Validate and estimate a run without model calls")
    plan.add_argument("dataset", type=Path)
    plan.add_argument("--model", required=True)
    plan.add_argument("--limit", type=int, default=100)
    plan.add_argument("--mode", choices=["conservative", "balanced"], default="balanced")
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--max-output-tokens", type=int, default=512)
    plan.add_argument("--prices", type=Path)
    plan.add_argument("--output", type=Path)
    review = sub.add_parser("score-reviews", help="Score completed blinded rubrics")
    review.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "optimize":
            request = OptimizationRequest.model_validate_json(args.request.read_text())
            print(Optimizer().optimize(request).model_dump_json(indent=2))
        elif args.command == "audit-dataset":
            report = audit_dataset(args.dataset)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "benchmark":
            report = run_offline(args.dataset, args.repeats)
            write_offline_report(report, args.output)
            print(json.dumps(report["summary"], indent=2))
        elif args.command == "plan-evaluation":
            prices = Prices.model_validate_json(args.prices.read_text()) if args.prices else None
            report = plan_summary(
                build_plan(
                    args.dataset,
                    model=args.model,
                    limit=args.limit,
                    mode=args.mode,
                    seed=args.seed,
                    max_output_tokens=args.max_output_tokens,
                    prices=prices,
                )
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "evaluate-online":
            if not os.getenv("OPENAI_API_KEY"):
                parser.error("OPENAI_API_KEY is required; no model calls were made")
            prices = Prices.model_validate_json(args.prices.read_text()) if args.prices else None
            if prices and prices.model != args.model:
                parser.error("Prices model must match the requested evaluation model")
            from tokenflow.llm.providers import OpenAIProvider

            summary = run_online(
                args.dataset,
                args.output,
                OpenAIProvider(args.model, max_output_tokens=args.max_output_tokens),
                mode=args.mode,
                limit=args.limit,
                prices=prices,
                seed=args.seed,
                resume=args.resume,
            )
            print(json.dumps(summary, indent=2))
        elif args.command == "score-reviews":
            result = evaluate_reviews(args.run)
            atomic_json(args.run / "quality-review.json", result)
            print(json.dumps(result, indent=2))
    except (BudgetExceeded, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
