# tests/index_search — AGENTS.md
Purpose: tests of `gpt2_demo_service/index_search/` against `fixture_expectations.json` (candidate and rerank counts of 6 or more, floor 0.15, threshold 0.45).

## Files
- `test_lexical_and_dense_search.py` — lexical and dense rankings equal the fixture, ties by row, dense top-k equals a full sort, empty queries.
- `test_rank_fusion.py` — the recorded fusion example, 1-based ranks, ties by id, empty input.
- `test_evidence_retriever.py` — fused order, best dense score and pair rankings equal the fixture; fg-5 and fg-6 get no evidence; pair threshold; single-search modes; reranker order and gate; head-only rerank; empty evidence records.
