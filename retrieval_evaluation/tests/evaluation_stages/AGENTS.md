# tests/evaluation_stages — AGENTS.md
Purpose: end-to-end tests of the stages on the fixture with fake service objects and a temporary results folder.

## Files
- `test_retrieval_and_calibration_stages.py` — the retrieval decisions and test run, the floor and the missing pair threshold, and how calibrated settings are built.
- `test_answers_and_report_stages.py` — the grid (four configurations without a pair threshold, six with one), the two final arms, the recommended settings with the gate lines, and the written report.
