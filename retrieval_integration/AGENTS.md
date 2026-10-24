# retrieval_integration — AGENTS.md
Purpose: the last package of the retrieval work (workflow `wf-1024-2000-got-rag`, DESIGN.md section 10.3): the scripts the integrator used to start the real service parts locally, to score a candidate model the same way as the evaluation, and to check the deployed Space as a visitor would (no token). Mac only; the scripts import the service, the encoders and the evaluation, and write private results.

## Files
- `local_retrieval_startup_check.py` — starts retrieval exactly as the Space does (private download at the pinned commit, real encoders, checksum validation, warm-up) and asks three questions; prints ids, scores and timing only (no book text).
- `evaluate_candidate_model.py` — scores a candidate weights file (for example the grounding fine-tune) on the evaluation's final items and seeds, with retrieval and with the plain prompt, so it compares like with like against `../retrieval_evaluation/results/answers.json`.
- `run_reranker_variant.py` — re-runs the evaluation's retrieval and calibration stages into `results/reranker_variant/` with the Space slowdown factor measured live (1 instead of the planning factor 3), so the ranking model's decision and its evidence floor match what the Space really costs.
- `upload_candidate_weights.py` — uploads a candidate weights file to the private weights repo under a NEW name (refuses an existing name and a public repo), so the live model file always stays for rollback.
- `compare_server_temperatures.py` — scores the live (grounding) model WITH retrieval at one server temperature on the fact-development questions, the 30 facts or the old mistakes (same seeds and word rules as the evaluation, 3 samples each); used on 2026-10-25 to choose the Space's default temperature.
- `live_space_verification.py` — token-free live check of the deployed Space: `/health`, the portfolio contract, the spam path (exact body, p95), the 30 fact questions one by one with timing, retrieval stage times from `Server-Timing`, memory before and after.

## Subfolders
- `results/` — private result files of these scripts (start-up check, reranker variant, candidate scores, live verification records). Not uploaded anywhere.

The final write-up of the whole retrieval work is `../docs/RESULTS.md`.
