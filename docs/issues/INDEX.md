# M1 issue plan

Repository: [MMM1122/tokenflow](https://github.com/MMM1122/tokenflow) (private).

All ten tasks are published. Seven engineering tasks are complete; three remain open for independent data, live evaluation, and the product decision.

- [#1: Define benchmark](https://github.com/MMM1122/tokenflow/issues/1) - open; [acceptance criteria](01-define-benchmark.md).
- [#2: Define input/output contracts](https://github.com/MMM1122/tokenflow/issues/2) - closed; [acceptance criteria](02-contracts.md).
- [#3: Tokenizer and measurements](https://github.com/MMM1122/tokenflow/issues/3) - closed; [acceptance criteria](03-tokenizer.md).
- [#4: Preservation policy](https://github.com/MMM1122/tokenflow/issues/4) - closed; [acceptance criteria](04-preservation.md).
- [#5: Document deduplication](https://github.com/MMM1122/tokenflow/issues/5) - closed; [acceptance criteria](05-deduplication.md).
- [#6: Relevance and importance ranking](https://github.com/MMM1122/tokenflow/issues/6) - closed; [acceptance criteria](06-ranking.md).
- [#7: Extractive compression](https://github.com/MMM1122/tokenflow/issues/7) - closed; [acceptance criteria](07-compression.md).
- [#8: Budget enforcement and regression tests](https://github.com/MMM1122/tokenflow/issues/8) - closed; [acceptance criteria](08-budget-and-tests.md).
- [#9: Paired evaluation and quality gate](https://github.com/MMM1122/tokenflow/issues/9) - open; [acceptance criteria](09-paired-evaluation.md).
- [#10: Reproducible package and go/no-go](https://github.com/MMM1122/tokenflow/issues/10) - open; [acceptance criteria](10-reproducibility.md).

The M2 local HTTP/Python preview is implemented. Production infrastructure remains behind the real quality gate.

Next priority: configure a model for paired evaluation on independent traces. Keep conservative mode as the default until evidence supports more aggressive strategies.
