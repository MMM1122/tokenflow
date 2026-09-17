import argparse
import json
import os
from pathlib import Path

from tokenflow.core.optimizer import Optimizer
from tokenflow.evaluation import (
    Prices,
    evaluate_reviews,
    run_offline,
    run_online,
    write_offline_report,
)
from tokenflow.models import BudgetExceeded, OptimizationRequest


def main():
    parser = argparse.ArgumentParser(description="TokenFlow context optimizer and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize", help="Optimize one JSON request without any LLM call")
    optimize.add_argument("request", type=Path)
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
    review = sub.add_parser("score-reviews", help="Score completed blinded rubrics")
    review.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "optimize":
            request = OptimizationRequest.model_validate_json(args.request.read_text())
            print(Optimizer().optimize(request).model_dump_json(indent=2))
        elif args.command == "benchmark":
            report = run_offline(args.dataset, args.repeats)
            write_offline_report(report, args.output)
            print(json.dumps(report["summary"], indent=2))
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
                OpenAIProvider(args.model),
                mode=args.mode,
                limit=args.limit,
                prices=prices,
            )
            print(json.dumps(summary, indent=2))
        elif args.command == "score-reviews":
            result = evaluate_reviews(args.run)
            (args.run / "quality-review.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    except (BudgetExceeded, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
