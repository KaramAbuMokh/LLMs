# retrieval_evaluation — AGENTS.md
Purpose: the offline evaluation of retrieval for the Game of Thrones model (workflow `wf-1024-2000-got-rag`, DECISIONS `hq.gpt2_got_rag_1024`, DESIGN.md section 11). It builds the question sets, measures retrieval, calibrates the two thresholds, runs the model with and without retrieval, applies the ship gate and writes the recommended settings and a report. Runs on the Mac only; nothing here ships to the Space and nothing here is uploaded.

## Rules
- Reads the frozen contract (`../gpt2_service_huggingface_space/retrieval_contracts/`) and nothing else shared. Only `run_evaluation_command.py` loads the Space service code (and the model); every stage and runner receives the service objects as arguments, so the unit tests run with fakes and the shared fixture.
- Never imports the index builder. The answer locator both need is in the contract.
- Books are copyrighted: no book text in tests, results or reports. Results hold ids, offsets, scores, counts and the model's own answers.
- Files stay under 150 lines, a comment on every statement, full-word names; `tests/test_package_rules.py` checks these and the import boundary.
- Run the tests from this folder with `python -m pytest` (needs numpy and pytest; venv `~/.venvs/got-retrieval-evaluation`). The real-data check runs with `-m data`.
- Run the command from the project folder with `PYTHONPATH=.:gpt2_service_huggingface_space`, in an environment that also has the service requirements: `python -m retrieval_evaluation.run_evaluation_command golden|retrieval|calibrate|answers|report|live|all`. `all` runs golden, retrieval, calibrate, answers and report in order; `live` is separate and only for after a deploy (one caller, at most 20 calls).

## Files
- `evaluation_settings.py` — every path and number of the evaluation in one place: data locations, block split, retrieval ablation rules, calibration targets, generation caps and sampling, and the ship gate numbers.
- `run_evaluation_command.py` — the command. Wires the service objects (settings, bundle loading, encoders, retriever, the model, the answer service) and runs the named stages. The only module that imports the service.
- `pytest.ini` — test settings (import paths, the `data` marker, which is skipped by default).
- `requirements.txt` — what the unit tests need, and a note on what real runs need.

## Subfolders
- `golden_sets/` — the question sets: held-out passage questions with their block split, the 30 fact questions, the new fact-development questions, the unanswerable questions, the regression questions.
- `metrics/` — pure scoring: ranking measures, passage relevance, word-overlap scores, copyright checks, the fact rules, answer-run scoring.
- `experiments/` — the runners and decision rules: retrieval ablation, threshold calibration, answer runs and their plan, the settings recommendation, the report writer.
- `evaluation_stages/` — one small module per command stage that joins the pieces above and writes the result files.
- `live_checks/` — contract check and latency probe for the deployed Space.
- `tests/` — unit tests on the shared fixture and fakes, plus the real-data check.
- `results/` — generated result files (question sets, ablation, calibration, answers, recommended settings, report).
