# M1 issue plan

Ten tasks track the original M1 acceptance criteria. Live evaluation and independent data requirements remain open. Completed engineering tasks can be closed after their evidence is published.

1. [Define benchmark](01-define-benchmark.md) - partially complete.
2. [Define input/output contracts](02-contracts.md) - engineering complete.
3. [Tokenizer and measurements](03-tokenizer.md) - engineering complete.
4. [Preservation policy](04-preservation.md) - engineering complete.
5. [Document deduplication](05-deduplication.md) - engineering complete.
6. [Relevance and importance ranking](06-ranking.md) - engineering complete.
7. [Extractive compression](07-compression.md) - engineering complete.
8. [Budget enforcement and regression tests](08-budget-and-tests.md) - engineering complete.
9. [Paired evaluation and quality gate](09-paired-evaluation.md) - partially complete.
10. [Reproducible package and go/no-go](10-reproducibility.md) - partially complete.

Additional implementation: the M2 local HTTP/Python preview validates integration. It does not authorize skipping the M1 quality gate or claim production readiness.

Next priority: run paired evaluation on independent traces. Keep conservative mode as the default until evidence supports more aggressive strategies.
