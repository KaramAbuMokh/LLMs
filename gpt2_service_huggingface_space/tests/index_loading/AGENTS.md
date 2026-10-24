# tests/index_loading — AGENTS.md
Purpose: tests of `gpt2_demo_service/index_loading/` against the shared fixture bundle.

## Files
- `test_bundle_loading.py` — the fixture loads; a changed byte, a wrong encoder fingerprint, wrong tokenizer or indexed-text versions, wrong counts and a missing manifest are each refused with a reason.
- `test_bundle_download.py` — the Hub call asks for the dataset snapshot at the pinned commit with only the bundle's files; unpinned revisions are refused before any network call.
