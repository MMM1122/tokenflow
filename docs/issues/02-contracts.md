# Define input/output contracts

Model queries, history, and retrieved evidence separately with explicit lossiness.

## Acceptance criteria

- [x] Validate fields, unique document IDs, roles, and request-size limits.
- [x] Expose protected, format, importance, input_budget, and mode controls.
- [x] Return complete messages, decisions, reasons, token measurement semantics, and a lossy flag.
- [x] Document source citations and pinning for occurrence-count semantics.

Evidence: tokenflow/models.py; tests/test_optimizer.py.
