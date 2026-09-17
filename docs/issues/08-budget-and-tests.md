# Budget enforcement and regression tests

Enforce budgets without silently dropping instructions.

## Acceptance criteria

- [x] Fit the complete estimated input budget or return structured BudgetExceeded.
- [x] Balanced mode removes lower-ranked unprotected context; conservative mode retains unique evidence.
- [x] Return HTTP 422 for impossible budgets before making any provider call.
- [x] Test Unicode, code, corrections, controls, protected text, and non-growing prompts.

Evidence: tokenflow/core/budget.py; tests/.
