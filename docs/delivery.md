# Delivery status and engineering decision

M1 offline engineering and an M2 local integration preview are implemented. The complete long-term roadmap has not been marked finished.

## Verified locally

- 87 pytest tests pass; Ruff lint and formatting checks pass.
- Editable package installation and the installed `tokenflow` CLI work.
- A real uvicorn loopback smoke test returned HTTP 200 from health and optimization endpoints. The example input decreased from 121 to 103 estimated tokens, retaining one of two identical documents.
- The smoke-test server was shut down afterward. No real model calls were made.
- The offline benchmark covers 100 synthetic cases from 20 families, measured three times per mode and case. See benchmarks/results/m1-offline/ for the latest complete results.

## Evaluation reliability update

The no-call preflight validates the entire dataset and estimates both evaluation arms with explicit optional prices. The v2 runner records intent before each provider request and saves checksummed results atomically. Resume skips completed calls, treats unfinished attempts as uncertain without retrying, rejects changed experiment identities, and retains reviewer edits. OS locking prevents cooperating local writers from executing the same run concurrently. Tests cover interruptions, lost result writes, process exit, corrupted artifacts, changed settings, and model snapshot drift. These are reliability checks using test doubles; no new live quality or billing evidence is claimed.

## Dataset and review integrity update

The dataset audit reports 92 distinct baseline inputs among 100 cases in 20 declared families. The no-document and no-redundancy control families each repeat an identical input five times. No exact cross-family or cross-split input duplication was detected. The original fixture and benchmark measurements are preserved; automated inspection cannot establish independent provenance.

Blinded scoring now verifies immutable pair content, family assignment, answer mappings, and summary values against the saved plan and checksummed journal. Results include dataset, plan, and review fingerprints. Tests reject altered evidence and verify that even perfect test-double scores leave the independent-data product gate pending. No live evaluation evidence was added.

## Interpretation

The initial run measured 37.81% mean input reduction in conservative mode and 41.16% in balanced mode. Median reductions were 41.37% and 43.73%. Warm p95 overhead was approximately 2.5 ms on the local machine; timing varies between runs.

All annotated required context and protected content survived, and zero-redundancy controls saved zero tokens. Counts are actual BPE measurements of local serialization, not provider billing tokens. High duplicate density limits generalization; retaining fixture labels does not prove answer quality. Weighted aggregate reduction and mean per-request reduction are different statistics and are both reported.

Lossy processing added only about 3.35 percentage points of mean reduction on this suite. Keep conservative mode as the default and prioritize real quality evaluation. These measurements support another experiment, not a claim of proven product value.

## Remaining work and reasons

- Real quality, billing, and model latency: no configured OPENAI_API_KEY/OPENAI_MODEL and no independent developer traces. Paired calls, failure records, usage/cost computation, blinded review, and scoring are implemented.
- Published the private [MMM1122/tokenflow](https://github.com/MMM1122/tokenflow) repository and synchronized ten English issues: seven closed, three open. A pinned GitHub Actions workflow verifies Python 3.11 and 3.12. See docs/issues/INDEX.md for acceptance criteria and status.
- History compression, semantic embeddings, and model-generated summaries remain deferred until a real failure set covers references, implicit constraints, cross-language queries, and multi-hop facts.
- Docker, PostgreSQL, cache, and routing remain behind the quality and demand gates. A local HTTP interface does not establish a need for production infrastructure.

## Known limitations

The gateway is a single-process local preview with bounded, non-persistent metrics. It lacks public-deployment rate limiting, tenant isolation, admission control, and streaming request-size enforcement. Document role separation is not a prompt-injection guarantee.

The local test run emits one upstream Starlette/AnyIO deprecation warning. Core logic, provider request construction with a test double, and real local HTTP behavior are verified separately. Live provider connectivity and answer quality remain unverified.

The user's preferences for autonomous authorized work and English project files are recorded in AGENTS.md.
