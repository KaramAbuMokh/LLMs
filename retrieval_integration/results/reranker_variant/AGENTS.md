# reranker_variant — AGENTS.md
Purpose: result files of the evaluation's retrieval and calibration stages, re-run by `../../run_reranker_variant.py` with the Space slowdown factor measured live (1.0 instead of the planning factor 3.0). The main results in `../../../retrieval_evaluation/results/` stay as the first run. Private; ids, scores and counts only.

## Files
- `golden_sets.json` — copy of the question sets (the same 816 / 189 / 196 / 30 / 42 / 40 / 4 split).
- `retrieval_ablation.json` — the ablation again, with the reranker now chosen (hit@3 gain 0.116, estimated Space p95 479 ms) and ONE run of the chosen retrieval on the test questions (hit@5 0.781, reciprocal rank 0.715).
- `calibration.json` — the evidence floor on the rerank score (0.102: 90% of development, 100% of fact-development, 37.5% of the unanswerable questions keep evidence) and the pair threshold (none).
