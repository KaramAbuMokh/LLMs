# results — AGENTS.md
Purpose: private result files of the integration scripts. They hold the model's own answers, ids, scores and timings, and at most one 25-word snippet per live answer; never book passages. Not uploaded.

## Files
- `local_retrieval_startup_check.json` — output of the local start-up check of the real retrieval path.
- `live_space_verification*.json` — records of the token-free checks of the deployed Space (health, contract, spam, the 30 fact questions with timing and sources). `live_space_verification.json` is the latest (after the final hardening publish `c304d51d`, temperature 0.3); `live_space_verification_before_hardening.json` is the check of commit `2fe874e6`.
- `temperature_*.json` — every generated answer and the scores of the temperature comparison of 2026-10-25 (`fd_t03/t05/t07` = 42 fact-development questions at 0.3, 0.5, 0.7; `g2g4_t03/t07` = the 30 facts and the 4 old mistakes at 0.3 and 0.7), written by `../compare_server_temperatures.py`.
- `temperature_live_space_comparison.json` — the same comparison done LIVE on the Space (30 facts, three passes at 0.3 and three at 0.7, one call per question and pass): 0.3 gave 18/14/17 right (majority 18), 0.7 gave 16/13/14 (majority 15).
- `live_space_verification_first_publish.json` — the same check on the first publish (live model, reranker off): 11 of 30.
- `live_latency_probe_final.json` — the 20-call latency probe on the final Space (one caller, warm).
- `candidate_grounding_model_with_reranker.json` — scores of the grounding fine-tune on the evaluation's final items and seeds, with retrieval (reranker, rerank floor) and with the plain prompt (written by `../evaluate_candidate_model.py`): 17 of 30 facts against 12.

## Subfolders
- `reranker_variant/` — the retrieval and calibration stages re-run with the live-measured Space cost (reranker chosen, rerank-score evidence floor 0.102, test hit@5 0.781).
