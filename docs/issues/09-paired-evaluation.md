# Paired evaluation and quality gate

Validate or falsify actual model quality and cost benefits.

## Acceptance criteria

- [x] Use identical model/settings and seeded randomized baseline/optimized call ordering.
- [x] Persist each call with usage and latency; include failures and disable automatic retries.
- [x] Accept explicit prices for cached input and output and separate experimental cost from savings.
- [x] Export blinded pairs and implement rubrics, family bootstrap, and critical-violation gates.
- [x] Validate the complete dataset and expose a no-call preflight with optional explicit cost estimates.
- [x] Resume local experiments using a durable journal without retrying uncertain or failed attempts.
- [x] Reject changed run identities, corrupt journals, concurrent writers, and model snapshot drift.
- [x] Bind reviewed answers and blinding mappings to the original plan and checksummed journal before scoring.
- [ ] Configure real model credentials and execute paired live evaluation.
- [ ] Complete independent blinded review and assess cost and latency on real traces.

Evidence: tokenflow/evaluation.py; tokenflow/experiments.py; tokenflow/run_store.py; tests/test_evaluation.py; tests/test_experiments.py; docs/live-evaluation.md.

Remaining dependency: independent, consented traces and live model evaluation. Offline proxies and mocked providers do not satisfy this requirement.
