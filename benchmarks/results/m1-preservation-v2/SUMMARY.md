# TokenFlow offline regression results

Generated: 2026-09-18T02:34:46.164620+00:00

100 synthetic cases / 20 scenario families. No live LLM calls. Each case is measured three times by default.

Token counts are local serialized-message BPE estimates, not billing usage.

## conservative

- Mean input reduction: 37.81%
- Median input reduction: 41.37%
- Weighted input reduction: 63.65%
- Warm overhead p95: 2.76648779399693 ms
- Required-context retention: 100.00%
- Protected-content retention: 100.00%
- Failures: 0 / 100

## balanced

- Mean input reduction: 41.16%
- Median input reduction: 43.73%
- Weighted input reduction: 66.15%
- Warm overhead p95: 2.97262673266232 ms
- Required-context retention: 100.00%
- Protected-content retention: 100.00%
- Failures: 0 / 100

## Decision

Engineering target met: True.

Product gate: **not evaluated**. Answer quality, billed cost savings and model latency are unmeasured. Do not claim the product maintains baseline answer quality.

- Synthetic related variants are not independent real conversations.
- Required-context retention is not answer-quality retention.
- Serialized BPE counts differ from provider billing tokens.
- High duplicate density favors deduplication; inspect control cases.

Per-case values, category/length/split breakdowns, dataset hash and environment versions are in offline.json. See docs/benchmark-protocol.md for acceptance rules.
