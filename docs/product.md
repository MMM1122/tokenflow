# TokenFlow product definition and milestone boundaries

Serve developers making substantial LLM API calls. Reduce duplicate and low-value context before inference, while separately measuring answer quality, total cost, and latency. The product automates context management instead of asking users to manually request shorter prompts.

## Success criteria

The north star is lower actual request cost subject to predeclared quality constraints. Input-token reduction is a leading indicator, not the outcome by itself. Include output usage, cached-input discounts, optimization overhead, errors, and answer quality.

M1 engineering acceptance is separate from product validation. Offline regression success justifies further experiments, not claims of answer equivalence. See benchmark-protocol.md for thresholds, measurement definitions, and limitations.

## Initial architecture

Caller -> validated request -> preservation policy -> document deduplication -> lexical relevance -> extractive selection -> token budget -> provider -> metrics.

- `tokenflow/models.py`: immutable request, document, decision, and result contracts.
- `tokenflow/core/`: optimizer independent of HTTP and provider SDKs.
- `tokenflow/llm/`: provider protocol, Responses adapter, and Python `TokenFlow.generate()`.
- `tokenflow/metrics/`: bounded in-memory metrics without prompts or answers.
- `tokenflow/evaluation.py`: offline measurements, paired calls, blinded review export, and scoring.
- `tokenflow/api/app.py`: M2 local integration preview.

## Input semantics and quality limits

Documents are retrieved evidence, where repeated same-source chunks normally have no multiplicity semantics. For counting, occurrence-level auditing, or occurrence-specific citations, mark the relevant chunks `protected=true`. Internal IDs support auditing; source names support citations.

Conservative mode removes only byte-identical, same-source, unprotected document duplicates. Even without dropping distinct information, stochastic model answers need not be identical.

Balanced mode permits lossy lexical near-duplicate filtering and paragraph selection. When needed, the budget stage drops unprotected documents using relevance, position-based recency, and caller importance. Recency assumes oldest-to-newest input order; it is not a verified timestamp. Approximate deduplication never crosses sources and has explicit comparison/length bounds.

All history roles, ordering, and content are preserved. Instructions and the current query are protected separately. Role separation and JSON wrappers are not a prompt-injection defense guarantee.

Lexical methods are weak on synonyms, cross-language references, multi-hop evidence, and implicit constraints. Heuristics cannot identify every essential fact. Callers must pin required evidence. Build a real failure set before selecting embeddings or state summaries.

## Milestones

1. M1 Optimizer: benchmark, implementation, preservation tests, and real quality experiments. Offline engineering is complete; live quality remains pending model credentials and independent data.
2. M2 Gateway: local preview implemented. Production work still needs authentication policy, admission control, raw-body limits, cancellation behavior, load testing, and integration evidence.
3. M3 Productionize: add Docker after quality and actual cost validation. Introduce PostgreSQL when persistent usage history requires a schema, migration, and retention policy.
4. M4 Intelligent Gateway: add caching and routing when traffic justifies them; apply quality and cost gates to each strategy.

No Kubernetes, multi-cloud, custom model training, complex agents, or ten-provider abstraction in the MVP. Operational requirements, rather than an arbitrary request count, determine the need for orchestration.

## Future cache keys

Include tenant, permissions, effective instructions, language, model/generation settings, context version, and tool state. Similar questions are not automatically interchangeable answers. Any semantic cache needs a separate false-hit evaluation.

## Data handling

Operational metrics retain only the latest 1,000 records in memory and disappear at process exit. Live evaluation directories intentionally contain model answers and reviewer inputs. They are explicit experiment artifacts, not content-free operational logs. Use consented, redacted data and exclude live runs from version control.

## Repository conventions

Use English for documentation, comments, commit messages, and issue text. Multilingual regression payloads are represented with Unicode escapes in source and fixture files so their behavior remains testable without non-English project prose.
