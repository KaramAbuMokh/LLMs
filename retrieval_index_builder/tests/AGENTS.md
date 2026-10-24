# tests — AGENTS.md
Purpose: unit tests of the index builder, one subfolder per area. They run offline in a few seconds. The shared fixture of the contract is the reference (the builder must reproduce it);
no test contains book text (invented sentences and invented vocabularies only). `conftest.py` points the GPT-2 vocabulary cache at the dataset folder so no test needs the network.
Run `python -m pytest` from `retrieval_index_builder/` (add `-m slow` for the real-model checks).

## Files
- `conftest.py` — keeps the tokenizer vocabulary offline.

## Subfolders
- `source_alignment/` — raw heading readers, heading patterns, chapter alignment, pair placement.
- `passage_chunking/` — sentence splitting and passage chunking.
- `build_pipeline/` — the command's arguments, the build report, part text loading, and one small whole build on invented books.
- `bundle_writing/` — conformance with the shared fixture, BM25 postings, upload safety.
- `encoders/` — tokenizer and padding tests with a tiny vocabulary, plus the slow real-model checks of the shared encoders.
