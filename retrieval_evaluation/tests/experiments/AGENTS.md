# tests/experiments — AGENTS.md
Purpose: tests of the runners and decision rules, using the fixture's recorded rankings.

## Files
- `test_retrieval_ablation.py` — hand-computed metrics, variants as settings replacements, one retriever per variant, the reranker comparison.
- `test_threshold_calibration.py` — the fixture floor separates fg-1 to fg-4 from fg-5 and fg-6, leave-one-out never returns the pair itself, threshold choice and halves.
- `test_answer_run_planning.py` — items, seeded subsets, the generation clock, halving to fit the caps.
- `test_answer_runs.py` — spies show one retrieval per question, one answer call per configuration and sample, shared evidence and identical seeds.
- `test_settings_recommendation.py` — the decision rules and the ship gate on hand-made numbers.
- `test_evaluation_report_writing.py` — tables, the inflation sentence, the margin, skipped stages, result files.
