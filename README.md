# TokenFlow

[Repository](https://github.com/MMM1122/tokenflow) | [Issues](https://github.com/MMM1122/tokenflow/issues) | [CI](https://github.com/MMM1122/tokenflow/actions/workflows/ci.yml)

Reduce redundant LLM context automatically and measure the tradeoffs in quality, cost, and latency.

**Status: M1 offline optimizer and evaluation tooling, plus an M2 local gateway preview.** Optimization, tests, and offline benchmarks require no API key. Live answer quality and billing savings remain unvalidated.

## Quick start

Requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
tokenflow optimize examples/request.json
pytest -q
```

`requirements.lock` records the dependency versions tested on Python 3.12/macOS, including development and gateway packages. It is not a cross-platform hash lock. Install only the core with `pip install -e .`, or resolve the supported ranges with `pip install -e '.[gateway,dev]'`.

The first tiktoken run downloads a public encoding table. For offline deployment, preload and preserve `TIKTOKEN_CACHE_DIR`. Subsequent optimization requires neither network access nor model calls. Character-count heuristics are never substituted for BPE tokenization.

## Python integration

```python
from tokenflow import Document, OptimizationRequest, Optimizer

request = OptimizationRequest(
    query="What does a Docker image contain?",
    documents=(
        Document(id="a", source="guide", content="An image contains the app and dependencies."),
        Document(id="b", source="guide", content="An image contains the app and dependencies."),
    ),
    mode="conservative",
    input_budget=8000,
)
result = Optimizer().optimize(request)
print(result.metrics)
print(result.decisions)
```

Every action has a reason. Conservative mode removes only byte-identical, same-source, unprotected document duplicates. Repeated user turns can express emphasis or corrections, so M1 retains the entire conversation. Mark essential evidence with `protected=True`; declare code, JSON, and tables with `format`. Balanced mode explicitly permits information loss. Its near-duplicate rule now accepts only horizontal whitespace differences within the same source; word order, case, punctuation, and paragraph changes remain distinct. Common negations receive heuristic protection, but essential evidence should still be explicitly pinned.

Documents represent a set of retrieved evidence. When occurrence counts matter, pin the affected chunks. Internal document IDs support auditing; model citations use `source`.

The budget applies to a BPE estimate of the complete serialized input, excludes output tokens, and is not a provider context-window guarantee. If protected content cannot fit, `BudgetExceeded` is raised. Instructions, the latest query, history, and protected blocks are never silently truncated.

## Offline benchmark

```bash
python benchmarks/build_fixtures.py
tokenflow audit-dataset benchmarks/synthetic-v1.jsonl
tokenflow benchmark benchmarks/synthetic-v1.jsonl --output benchmarks/runs/offline
```

The suite contains 100 synthetic cases from 20 scenario families with five context-size variants each, not 100 independent real conversations. It covers question answering, code, documents, multilingual input, corrections, constraints, long histories, and zero-redundancy controls. Multilingual test payloads use Unicode escapes in source and fixture files; all project prose is English. The [dataset audit](benchmarks/results/m1-offline/dataset-audit.json) identifies 92 distinct baseline inputs: each of the two no-document/no-redundancy control families repeats one input five times. No exact cross-family or cross-split input duplicates were found. These controls remain in the original suite; they are not independent observations.

See the [latest measured results](benchmarks/results/m1-preservation-v2/SUMMARY.md), [per-case data](benchmarks/results/m1-preservation-v2/offline.json), [original baseline](benchmarks/results/m1-offline/SUMMARY.md), and [predeclared protocol](docs/benchmark-protocol.md). High duplicate density favors deduplication. Required-fact retention is not answer quality. Offline quality, cost, and model-latency fields remain `null`.

## Paired live evaluation

Set `OPENAI_API_KEY` in the shell without adding the key to source control. Explicitly select a model available to your account; no model availability or price is assumed.

```bash
export OPENAI_MODEL='your-model-id'
tokenflow plan-evaluation benchmarks/synthetic-v1.jsonl \
  --model "$OPENAI_MODEL" --limit 100 --mode balanced
tokenflow evaluate-online benchmarks/synthetic-v1.jsonl \
  --model "$OPENAI_MODEL" --limit 100 --mode balanced \
  --output benchmarks/runs/live-first
```

The planner validates every dataset row and estimates both arms without an API key or model calls. Its optional price estimate is not a billing cap or confirmation of model access. Live evaluation can make 200 paid calls. It disables automatic retries, defaults to `--max-output-tokens 512` and `--seed 42`, uses a 30-second provider timeout, and sets `store=False`. Execution order and blinded answer placement are fixed before the first call.

To resume an interrupted run, repeat the same command and add `--resume`. The runner locks the local directory, checks dataset/configuration/code hashes, and skips completed attempts. Attempts with no durable result become `uncertain` and are never automatically repeated, because the provider may already have charged them. Failures remain in the denominator. See the [live evaluation runbook](docs/live-evaluation.md) for data preparation, artifacts, recovery, and review.

Optional `--prices path/to/prices.json` accepts `input_per_million`, `cached_input_per_million`, `output_per_million`, `currency`, `as_of`, and `model`. Without prices, monetary fields stay null. Failed calls may still incur charges, so runs with failures do not claim a complete experimental cost. Cached-input discounts and output length affect comparisons. If successful responses identify different model snapshots, the runner withholds cost/latency comparisons and rejects quality scoring.

Give reviewers `blind-review.jsonl`, but keep `review-key.jsonl` hidden. Each side needs four integer scores from 0 to 4:

```json
{"correctness": 4, "completeness": 4, "instruction_adherence": 4, "grounding": 4}
```

This illustrates the schema, not measured grades. Use 0 for unusable, 1 for major errors, 2 for partial correctness, 3 for minor defects, and 4 for fully satisfying the task. Complete `left_scores`, `right_scores`, and the boolean `left_critical_violation` / `right_critical_violation` fields, then run:

```bash
tokenflow score-reviews benchmarks/runs/live-first
```

The evaluator verifies the answers, original requests, blinding key, and summary against the saved plan and checksummed journal before scoring. Editing pair content or the answer mapping causes an error; edit only scores and violation fields. The evaluator weights scenario families equally, bootstraps paired family-level differences, and checks critical violations. Its output records dataset, plan, and review fingerprints. Passing a synthetic sample still does not satisfy the independent real-data product gate.

## Local HTTP preview

```bash
uvicorn tokenflow.api.app:create_app --factory --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/v1/optimize \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Open `http://127.0.0.1:8000/docs` for API exploration. `/health` and `/v1/optimize` never call a model. `/v1/generate` and `/v1/metrics` require a Bearer token matching `TOKENFLOW_API_KEY`. Generation also requires `OPENAI_API_KEY` and `OPENAI_MODEL` in the server environment. Variables are read at startup; `.env.example` is documentation and is not loaded automatically.

For live Python generation:

```python
import os
from tokenflow.llm.client import TokenFlow
from tokenflow.llm.providers import OpenAIProvider

client = TokenFlow(OpenAIProvider(os.environ["OPENAI_MODEL"]))
answer = client.generate(request)
print(answer.response.text)
```

This is a local development gateway, without production rate limits, persistence, or tenant isolation. Metrics retain the last 1,000 successful operations per process and exclude prompt/answer content. Optimization responses contain context and must be handled as application data. Role separation and JSON wrapping do not guarantee prompt-injection resistance.

## Development and next milestones

```bash
ruff check .
ruff format --check .
pytest -q
```

GitHub Actions runs lint, tests, deterministic fixture checks, and offline evaluation without model credentials. Performance measurements are recorded rather than compared to a noisy shared-runner latency threshold.

[Product and architecture](docs/product.md) | [Issue plan](docs/issues/INDEX.md) | [Delivery notes](docs/delivery.md)

Next, evaluate consented independent developer traces with paired live calls. Investigate cross-language, multi-hop, and history-state failures before introducing embeddings or generated summaries. Docker, PostgreSQL, caching, and routing follow validated needs. Kubernetes is outside the MVP.

## Primary implementation references

- [Responses create API](https://developers.openai.com/api/reference/python/resources/responses/methods/create): input, storage, truncation, and usage contracts.
- [tiktoken](https://github.com/openai/tiktoken): BPE encoding; local serialized counts do not replace provider-reported billing usage.
- [GitHub Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python): supported workflow structure.

This project does not claim affiliation with OpenAI or any platform provider.
