# TokenFlow M1 — benchmark protocol v1

Written before the first measured run. Target: developers sending text, conversation history and retrieved documents to an LLM. Optimize input cost subject to answer quality and latency constraints.

## Dataset and isolation

- 100 synthetic cases: 20 scenario families × 5 context-size variants. Include Q&A, code, documents, Chinese, cross-turn corrections, negation, exact duplicate documents, irrelevant context, and zero-redundancy controls.
- These are regression fixtures, **not 100 independent real conversations**. Variants within each family are correlated. Do not claim generalization or benchmark wins on real traffic.
- Fixture labels (`required_context`, answer checks) are evaluation-only. The optimizer receives only the request. Preserve dataset SHA256, configuration and package versions in reports.
- `dev` families support implementation debugging; `holdout` families support a first smoke check. The synthetic holdout is still authored in this repository, not a sealed external validation set. A shipping decision requires unseen, consented developer traces with independent review.
- Test conservative and balanced modes against the identical unoptimized serialized request. No unequal prompt wrappers or hidden baseline truncation. Report failures in the denominator; do not exclude difficult cases.

## Measurements

1. Input-token reduction: (before − after) / before; report arithmetic mean, median, weighted aggregate, categories, lengths and modes. Tokenizer counts use the exact canonical serialized messages, **not provider billing tokens**; actual usage is recorded separately online.
2. Warm optimization overhead: per-case median across 3 runs, plus p50/p95 across cases. Tokenizer initialization/download excluded and reported separately. End-to-end LLM latency only measured in paired online runs.
3. Protected-content and required-fact retention: deterministic regression checks; **not answer quality**.
4. Paired answer evaluation: same model/settings, baseline and optimized for each case; seeded randomized execution order to reduce ordering/cache bias; provider failures and incomplete outputs count as failures. Capture input, cached-input and output token usage, duration and task-check scores. No automatic retry that silently changes billing.
5. Answer-check score: fraction of explicit expected substrings found, with forbidden-answer checks. A narrow diagnostic that can be gamed; **not a general quality-retention score**. Human review must assess correctness, completeness, instruction adherence and factual grounding on 0–4 scales, with blinded left/right answers. Never execute model-generated code for scoring.
6. Cost: only compute online from user-supplied, dated input/cached-input/output prices. Report full paired experimental cost separately from per-request optimized savings. Output length and provider cache discounts matter. Without prices/real calls, cost and response-latency metrics are null, not inferred from input reduction.

## Predeclared progression gates

- Functional: all protected content preserved byte-for-byte; no prompt grows; budget met or a structured budget error is returned; no silent truncation. Regression suite passes.
- Engineering target: balanced mode mean input reduction ≥20% on this synthetic suite and warm p95 overhead ≤100 ms on the recorded local machine. These are engineering targets, not evidence of product value.
- Product gate: real paired model runs and blinded human evaluation on independent traces. Mean optimized quality must be ≥95% of baseline, paired mean normalized-score difference lower 95% confidence bound ≥−0.05, baseline mean ≥0.80, and no critical constraint violation. Bootstrap by independent conversation/family, not the 100 correlated variants. Review subgroup regressions, cost and p95 end-to-end latency before proceeding.
- **Offline results cannot pass the product gate.** Do not expand into production infrastructure just because a synthetic token-reduction target passed.

## Known corrections to the original plan

- Repeated user instructions may express emphasis or correction. Preserve all conversation messages in M1; safely compressing long histories requires a separate state/constraint evaluation.
- Default conservative mode only removes byte-identical duplicate document chunks with the same source. Balanced mode additionally uses lexical relevance, near-duplicate filtering and extractive paragraph selection, explicitly accepting potential information loss. Neither is semantic similarity or an LLM summary.
- Protect explicit `protected` blocks, non-prose formats, numeric/constraint-bearing passages and referenced sources. Heuristics are incomplete: callers must mark facts that must survive.
- An impossible budget fails explicitly instead of deleting protected instructions. A local estimate is not a provider context-window guarantee.
- “Explain Docker” and “Explain Docker in Chinese” cannot share an answer cache key. Any future cache must include language, all effective instructions, model/settings, context version and tenant scope.
- Changes to a cached prompt prefix may reduce provider cache discounts; fewer raw input tokens do not guarantee lower cost.

## Reproduction

See README for commands. Freeze this protocol before running a new real evaluation; report any later changes, don't tune thresholds after seeing results.

## Runner v2 amendment: experiment integrity

Added before any live evaluation. The original quality thresholds, fixture labels, and offline results above are unchanged.

- Validate every dataset row, including rows outside the selected subset, before a provider request. Require nonempty, unique answer checks and reject conflicting labels or required context absent from the original input.
- Precompute seeded case order, arm order, blinded answer placement, and optimized messages. Persist the plan and its checksum before execution. Selection and blinding do not depend on provider outcomes.
- Record each attempt durably before calling the provider. Resume only the same dataset bytes, model, output cap, mode, seed, limit, prices, tokenizer version, provider type, and Python source hash. Cooperating processes lock a local run directory.
- Never automatically repeat completed, failed, or uncertain attempts. An interrupted attempt without a durable result is uncertain even when it may not have reached the provider. Include it as a failure; report unknown cost as unknown. This is conservative recovery, not a provider-side exactly-once guarantee.
- Keep automatic task checks separate from blinded human quality scores. Preserve completed reviewer edits on export replay. Reject comparisons across different returned model snapshots.
- Preflight costs estimate uncached serialized input plus configured output caps for both arms. They are not measured bills or enforceable spending limits. Paid execution requires explicit model selection and credentials.
- Version 1 experiment directories cannot resume under v2. Existing offline result files remain valid records of their original run. See live-evaluation.md for the recovery procedure.

## Dataset audit and review integrity amendment

Added before any live evaluation; acceptance thresholds remain unchanged. A mechanical dataset audit now distinguishes case count, declared family count, and distinct rendered baseline inputs. The original synthetic suite contains 92 distinct inputs: two control families each contain five identical variants. Preserve and disclose these controls; do not count them as independent evidence. No exact duplicate inputs span families or splits in this fixture. Exact matching cannot prove independence or detect all shared origins.

Before scoring, verify the immutable reviewed content and blinded mapping against the persisted plan and checksummed attempt records, and recompute the run summary. Reject changed answers, requests, family assignments, keys, or mismatched summaries. Sort families before seeded bootstrap so row reordering does not change the calculation. Record dataset, plan, and review fingerprints with scores. Checksums provide consistency checks, not authentication. The product gate still requires independently reviewed provenance, real quality, cost, and latency evidence.
