# retrieval_contracts/shared_fixture/fixture_bundle — AGENTS.md
Purpose: a complete retrieval bundle in the real format, built from the invented two-book fixture corpus with the hashing fake encoder.
The index builder must reproduce these files from the same records (conformance test); the service and the evaluation load it as their test index.
Generated data, one thing: do not hand-edit (checksums in the manifest would stop matching). Kept whole on purpose although it has 8 data files.

## Files
- `manifest.json` — format version 1, index version (by the rule of DESIGN.md §6.3: build date plus a checksum over all seven data files), fake-encoder fingerprint, chunking and lexical settings, counts (6 passages, 4 pairs, 4 chapters) and a checksum per file.
- `book_passages.jsonl` — the 6 invented passages (part, book, chapter, character span, GPT-2 token count, text).
- `book_passage_vectors.npy` — 6 x 256 unit-length vectors of the indexed passage text (heading + text).
- `book_passage_lexical_index.npz` — lexical postings of the indexed passage text: sorted terms, posting starts, posting rows, precomputed BM25 weights.
- `question_answer_pairs.jsonl` — the 4 invented question-answer pairs (one unlocated).
- `question_vectors.npy` — 4 x 256 vectors of the pair questions alone.
- `question_answer_lexical_index.npz` — lexical postings of question + answer.
- `chapters.jsonl` — the 4 invented chapter spans (spans start after the inline heading word).
