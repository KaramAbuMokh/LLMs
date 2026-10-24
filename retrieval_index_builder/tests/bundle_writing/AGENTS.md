# tests/bundle_writing — AGENTS.md
Purpose: tests of `bundle_writing/`.

## Files
- `test_fixture_conformance.py` — builds a bundle from the shared fixture's records and checks lexical arrays (exact), vectors (1e-6), record files (same bytes), the manifest's own check and the fixture index version rule.
- `test_lexical_postings.py` — BM25 weights equal a naive loop on random corpora, array layout rules, empty corpus, same top three as rank_bm25 for one-term queries.
- `test_bundle_upload.py` — a mocked Hub client: a public repo is refused with nothing uploaded, a private repo gets one commit and a receipt, a damaged bundle never reaches the Hub, the token is never in results.
