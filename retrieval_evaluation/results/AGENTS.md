# results — AGENTS.md
Purpose: files the evaluation command generates. Private, never uploaded, never committed. They hold question ids and questions, gold offsets, scores, counts and the model's own answers; never passage text.

## Files
- `golden_sets.json` — every question set with its counts (written by the golden stage; built on the real data: 816 answerable, 443 located, 58 near a block edge, 189 development, 196 test, 30 facts, 42 fact-development, 40 unanswerable, 4 regression).
- `retrieval_ablation.json` — the retrieval ablation on the development questions, the chunk-size comparison, the three decisions and ONE run of the chosen retrieval on the test questions (run on 2026-10-24 against the real bundles: hybrid search, 128-token passages, reranker off by the Mac-time rule).
- `calibration.json` — the evidence floor (dense score 0.636) with its trade-off table, and the pair-example threshold (none reached the precision target).
- `answers.json` — every generated answer of the answers stage (1,856 generations of the live model: the four layouts-and-budgets grid, then the baseline and the chosen arm once on the reported sets).
- `recommended_settings.json` and `EVALUATION-REPORT.md` — the ship gate and the report of that run (prompt-only retrieval with the live weights: the grounding fine-tune was needed).
Later runs of the integrator (the reranker variant with the live-measured slowdown, the grounding model) are in `../../retrieval_integration/results/`; after a deploy `live_checks.json` may be added here.
