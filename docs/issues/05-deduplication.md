# Document deduplication

Remove repeated retrieval evidence without deleting dialogue emphasis or source provenance.

## Acceptance criteria

- [x] Conservative mode removes only identical, same-source, same-format unprotected documents.
- [x] Preserve provenance and audit relationships without whitespace/code rewriting.
- [x] Make near-duplicate filtering opt-in and explicitly lossy; permit only horizontal whitespace differences after preservation checks.
- [x] Retain highly similar claims with changed roles, words, punctuation, operators, or case.
- [x] Bound approximate alignment length and comparisons; retain context when limits are reached.

Evidence: tokenflow/core/deduplicator.py; tests/test_optimizer.py.
