# Define benchmark

Freeze the evaluation protocol before optimizing for its measurements.

## Acceptance criteria

- [x] Create 100 synthetic cases across 20 correlated scenario families, including multilingual, code, document, multi-turn, and control scenarios.
- [x] Separate labels from optimizer input; record dataset hash, mode, environment, and counting convention.
- [x] Keep unmeasured quality, cost, and model latency null; retain failures in denominators.
- [x] Audit exact input duplication, family/split overlap, and conflicting answer checks without claiming independent provenance.
- [ ] Add consented, redacted, independent developer traces and freeze the real evaluation set.

Evidence: benchmarks/synthetic-v1.jsonl; benchmarks/results/m1-offline/dataset-audit.json; docs/benchmark-protocol.md; tokenflow/dataset_audit.py.

Remaining dependency: independent, consented traces and live model evaluation. Offline proxies and mocked providers do not satisfy this requirement.
