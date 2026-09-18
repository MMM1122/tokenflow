# Reproducible package and go/no-go

Make M1 reproducible and support an honest product decision.

## Acceptance criteria

- [x] Provide an installable package, CLI, dependency snapshot, examples, and local API preview.
- [x] Publish per-case and grouped results, protocol, and limitations.
- [x] Avoid paid calls by default and exclude prompt/answer content from operational metrics.
- [x] Verify tests, lint, installed commands, and local HTTP behavior.
- [x] Verify local HTTP failure responses, correlated request IDs, bounded outcome history, and explicit unknown billing usage.
- [ ] Pass the independent real-quality gate before starting production infrastructure.

Evidence: README.md; docs/delivery.md; benchmarks/results/m1-offline/.

Remaining dependency: independent, consented traces and live model evaluation. Offline proxies and mocked providers do not satisfy this requirement.
