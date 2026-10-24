# tests — AGENTS.md
Purpose: unit tests for the evaluation. They run on the shared fixture and on fakes, with no model, no network and no book text. One subfolder per package folder.

## Files
- `test_package_rules.py` — package rules as tests: no file over 150 lines, only the command imports the service at run time, no forbidden abbreviations or one-letter names, a comment on every statement.

## Subfolders
- `golden_sets/` — held-out split on the fixture, fact-development rules, the other question sets and the JSON round trip.
- `metrics/` — hand cases for the ranking measures, relevance, overlap scores, copyright checks, fact rules and answer scoring.
- `experiments/` — ablation, calibration, answer runs and their plan, decision rules, report writing.
- `evaluation_stages/` — retrieval, calibration, answers and report stages end to end on the fixture.
- `live_checks/` — the live checks against a local fake Space.
- `real_data/` — one check on the real books and pairs (marker `data`, skipped by default).
- `support/` — shared test doubles and fixture-based inputs.
