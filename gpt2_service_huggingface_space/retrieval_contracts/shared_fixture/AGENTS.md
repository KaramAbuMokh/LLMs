# retrieval_contracts/shared_fixture — AGENTS.md
Purpose: the ONE tiny fixture the index builder, the service and the evaluation all test against, so each can be built and tested alone.
Everything is invented (two made-up books, "The Glass Coast" and "The Salt Tower"); there is no book text here. Never shipped to the Space
(the publish script ignores this folder). Generated once by the architect with the formulas in DESIGN.md §6; do not hand-edit the data files.
If a package's code disagrees with a number here, report it to the design owner instead of changing the fixture.

## Files
- `__init__.py` — package marker.
- `fixture_files.py` — paths to everything below and loaders for the part texts, the golden questions (with gold character spans) and the expectations.
- `fixture_encoders.py` — two deterministic fakes: a hashing text encoder (lexical tokens hashed into 256 columns, unit length; its fingerprint is in the fixture manifest) and a token-overlap pair scorer (stand-in reranker).
- `fixture_part_texts.json` — the two invented part texts the passages were cut from, with inline chapter headings like the cleaned books.
- `fixture_golden_questions.json` — six questions: four answerable (gold spans, short answer, accepted words) and two the corpus cannot answer.
- `fixture_expectations.json` — the numbers every package must reproduce: lexical tokens of two sample sentences, the lexical (BM25) settings, a rank-fusion example, the fixture evidence floor (0.15) and pair threshold (0.45), and per question the dense, lexical and fused passage rankings, the best dense score, the pair rankings and the stand-in rerank scores. Rankings were computed with every passage as a candidate (candidate counts at least 6).
- `test_shared_fixture.py` — self-test of the contract against this fixture (manifest round trip, checksums, spans, tokens, vectors, pair-offset mapping, strict parsing, chapter labels, and the answer locator placing two pair answers inside their gold sentences).

## Subfolders
- `fixture_bundle/` — the reference bundle in the real format (manifest plus seven data files) built from the invented corpus with the hashing encoder.
