# tests/live_checks — AGENTS.md
Purpose: tests of the live checks against a small fake Space served from a local thread.

## Files
- `test_live_checks.py` — a good Space passes, each contract break is reported, the probe reads Server-Timing and refuses more than 20 calls.
