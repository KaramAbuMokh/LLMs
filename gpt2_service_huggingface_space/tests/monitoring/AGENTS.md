# tests/monitoring — AGENTS.md
Purpose: tests of `gpt2_demo_service/monitoring/`.

## Files
- `test_stage_timing.py` — named stages, repeated stages add up, a failing stage is still timed, callers get a copy.
- `test_request_logging.py` — the log line has exactly the documented keys and carries no question, answer or snippet text.
