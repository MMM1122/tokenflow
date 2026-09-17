# Relevance and importance ranking

Provide an explainable deterministic selection order for the budget stage.

## Acceptance criteria

- [x] Use IDF-weighted query coverage with English terms and Chinese character bigrams.
- [x] Score using 0.65 relevance + 0.10 positional recency + 0.25 caller importance.
- [x] Select documents without reordering remaining history or evidence.
- [x] Document positional-recency assumptions and cross-language/multi-hop limitations.

Evidence: tokenflow/core/relevance.py.
