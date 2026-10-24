# tests/encoders — AGENTS.md
Purpose: tests of the shared encoders in `../../../gpt2_service_huggingface_space/retrieval_encoders/` (their tests live here because the service's own tests folder belongs to another package).

## Files
- `test_word_piece_tokenizing.py` — tokenizer limit checks, truncation of single texts and pairs, per-batch padding, input selection; uses a tiny invented vocabulary, no model files.
- `test_pinned_encoders_slow.py` — marked `slow`, needs the downloaded models: fingerprint, unit rows in input order independent of batching, the reranker's input cap, and (with sentence-transformers installed) parity with the reference library.
