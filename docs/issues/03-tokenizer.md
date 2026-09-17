# Tokenizer and measurements

Make counting reproducible without confusing estimates with billing.

## Acceptance criteria

- [x] Use BPE over the identical canonical message serialization for both arms.
- [x] Handle multilingual text and special-token-like literal input.
- [x] Reject unsupported encodings instead of silently using character-count heuristics.
- [x] Record provider input, cached-input, and output usage separately.

Evidence: tokenflow/core/tokenizer.py; tokenflow/llm/providers.py.
