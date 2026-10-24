# index_search — AGENTS.md
Purpose: finds evidence for one question in a loaded bundle. Needs numpy and the contract only (plus the standard-library stage timer): importing
`evidence_retriever` loads no torch, pydantic or fastapi (tested in a fresh process). Encoders arrive as objects through the contract's interfaces.

## Files
- `evidence_search_settings.py` — every number the search uses (mode, candidate counts, fusion constant, rerank switch and count, evidence floors, passage and example limits, the `pair_examples_enabled` switch and the pair threshold), declared without defaults.
- `evidence_records.py` — what retrieval returns: the routes, scored passages and pairs, the retrieved-evidence record, the empty result for "retrieval off or failed", the stage names, and the helper that names the route from what a set of evidence holds.
- `lexical_search.py` — scores documents by summing the precomputed BM25 weights of the distinct query words; top matches best first, ties by row.
- `dense_search.py` — exact inner-product search over unit vectors (the score is the cosine), ties by row, exact at the cut-off.
- `rank_fusion.py` — reciprocal rank fusion of several ranked id lists, 1-based ranks, ties by id.
- `evidence_retriever.py` — the retriever: encodes the question once (with and without the passage instruction), searches and fuses passages and pairs, optionally reranks the best candidates, applies the one evidence gate (best dense cosine, or best rerank score) and, only when `pair_examples_enabled` is true, searches the pairs and applies the pair threshold; times each stage.
