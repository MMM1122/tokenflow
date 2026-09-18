# Preservation policy

Protect instructions and essential evidence before pursuing token savings.

## Acceptance criteria

- [x] Preserve instructions, current query, and all history content and ordering.
- [x] Honor explicit protection, structured formats, constraint heuristics, and referenced sources.
- [x] Cover emphasis, budget corrections, and multilingual constraints in regressions.
- [x] Document heuristic limits and defer history summaries to a separate evaluation.

Evidence: tokenflow/core/classifier.py; tests/test_optimizer.py.

Regression update: protect common English negations, including no/neither/nor and straight or typographic-apostrophe negative contractions. Under an impossible budget, these documents cause an explicit error rather than silent removal. See tests/test_optimizer_adversarial.py. Heuristics remain incomplete; caller-protected evidence is the explicit contract.
