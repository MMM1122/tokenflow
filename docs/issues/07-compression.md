# Extractive compression

Evaluate low-overhead original-text selection before adding another model call.

## Acceptance criteria

- [x] Only balanced mode selects paragraphs, retaining the opening and high-relevance passages.
- [x] Generate no new facts and accept changes only when the serialized prompt shrinks.
- [x] Keep content unchanged without a relevance signal; leave explicit budget selection separate.
- [x] Mark extraction as lossy and do not advertise it as semantic summarization.

Evidence: tokenflow/core/compressor.py; tests/test_optimizer.py.
