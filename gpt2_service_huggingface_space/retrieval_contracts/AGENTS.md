# retrieval_contracts — AGENTS.md
Purpose: the FROZEN contract between the three retrieval packages (offline index builder, the Space service, the evaluation): the record types,
the bundle file layout and manifest, the lexical word splitter, how records become indexed text, the book-part table and the encoder interface.
Standard library only, so every package can import it. It ships inside the Space image because the service reads bundles with it.
Design: `../../docs/DESIGN.md` (section 6). Written by the architect of workflow `wf-1024-2000-got-rag`.

## Rules
- Frozen: builders do not edit these files. A needed change goes back to the design (DESIGN.md "Change log" section), then every
  version constant it touches is bumped (`BUNDLE_FORMAT_VERSION`, `LEXICAL_TOKENIZER_VERSION`, `INDEXED_TEXT_VERSION`) and the index is rebuilt.
- No book text in here and never in tests; the fixture is an invented two-book corpus.
- Self-test: `python -m pytest retrieval_contracts` from the service folder (needs numpy and pytest).

## Files
- `__init__.py` — package marker.
- `bundle_records.py` — the three stored record types (book passage, question-answer pair, chapter), the evidence kinds, and the strict JSON-lines writer and reader (unknown or missing keys fail loudly).
- `bundle_manifest.py` — bundle format version, the eight file names, the manifest (what the index was built from, with which encoder, chunking and lexical settings, a checksum per file), and the check that a folder matches its manifest.
- `book_parts.py` — the seven book parts (six training parts plus the held-out stretch of A Feast for Crows as part 6), their titles and publication order, and the mapping from a pair's coarse book location into a part.
- `lexical_tokens.py` — the one word splitter for lexical search, used both when the index is built and when a question is searched (stopwords, possessive and plural endings), with its version number.
- `indexed_text.py` — the heading put in front of each passage before it is embedded and lexically indexed ("A Storm of Swords, Tyrion (chapter 12): ..."), the text a pair is indexed with, and the chapter label shown in public citations.
- `rare_shingle_location.py` — the one rule for where a question-answer text sits in the book parts: rare 4-word runs (at most 3 in all parts) of the text, and their densest cluster of two or more hits. Used by the index builder to give pairs a book and chapter and by the evaluation for the gold places and relevance of held-out questions, so both agree. Offsets point into the original part text.
- `encoder_contracts.py` — the two pinned public encoders (dense text encoder and optional reranker, exact Hub commits), the passage-search instruction, the fingerprint the manifest stores, and the two calling interfaces the real encoders and the test fakes both satisfy.

## Subfolders
- `shared_fixture/` — the tiny shared test fixture every package tests against: an invented corpus as a complete reference bundle, golden questions, expected numbers, fake encoders and the contract self-test.
