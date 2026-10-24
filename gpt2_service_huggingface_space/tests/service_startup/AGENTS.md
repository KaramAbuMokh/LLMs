# tests/service_startup — AGENTS.md
Purpose: tests of `gpt2_demo_service/service_startup/` on the invented fixture bundle.

## Files
- `test_retrieval_startup.py` — a good bundle gives a ready status with counts, short revision and the number of book runs for the copy guard; a changed byte, a wrong encoder and an unexpected error each turn retrieval off with a reason that holds file names, numbers or an error type but never text.
