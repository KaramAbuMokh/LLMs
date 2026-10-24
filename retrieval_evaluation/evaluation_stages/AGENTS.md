# evaluation_stages — AGENTS.md
Purpose: one small module per stage of the command. Each joins the pure pieces (question sets, metrics, runners, rules), takes the service objects through one runtime record, and writes its result file into `results/`. No stage imports the service.

## Files
- `evaluation_runtime.py` — the record of service objects a real run needs: default search settings, a retriever factory, the bundle folders, the answer service, the request and generator makers, the layouts and a progress printer.
- `golden_sets_stage.py` — builds all question sets from the real data, fails if the fact-development checks find a problem, writes `golden_sets.json`, and loads it again for later stages.
- `retrieval_stage.py` — ablation on the development questions, the three retrieval decisions (search mode, reranker, chunk size), then one run of the chosen retrieval on the test questions; writes `retrieval_ablation.json`.
- `calibration_stage.py` — evidence floor (with its trade-off table) and pair threshold for the chosen retrieval; writes `calibration.json` and builds the calibrated search settings the answer runs use.
- `answers_stage.py` — the selection grid on fact-development and development questions, then the baseline and the chosen configuration once on the 30 facts, the regression set and the test questions; writes `answers.json`. When calibration found no pair threshold, the two layouts that need pair examples are skipped (they would repeat the baseline or the passage prompt) and the plan says so.
- `report_stage.py` — applies the ship gate and the fine-tune rule to the stage results and writes `recommended_settings.json` and `EVALUATION-REPORT.md`.
