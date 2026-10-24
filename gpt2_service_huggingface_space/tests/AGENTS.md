# tests — AGENTS.md
Purpose: unit and API tests of the Space service (`gpt2_demo_service/`), one subfolder per package. Run from the service folder with
`python -m pytest` (the frozen contract's 11 self-tests in `retrieval_contracts/` run in the same session). No test needs a network,
a model file or any book text; the only data are the invented shared fixture (`retrieval_contracts/shared_fixture/`) and generated words.

## Files
- `conftest.py` — fixtures every test folder shares: the GPT-2 tokenizer, a tiny random GPT-2, the fixture bundle loaded once, a copy of it with one changed byte, and the switch that skips the slow tests unless `GOT_RUN_SLOW_TESTS=1` is set.

## Subfolders
- `language_models/` — decoding cache, cached versus full forward pass, generator stop rules, sampling, and the slow real-weights test.
- `index_loading/` — bundle download arguments and the bundle loader's refusals.
- `index_search/` — lexical, dense and fused search and the evidence retriever against the fixture expectations.
- `grounded_answers/` — prompt layouts, packing, citations, the verbatim guard and the answer service.
- `monitoring/` — stage timer and the per-answer log line.
- `service_startup/` — retrieval start-up: ready status, the copy guard's index, every failure turning retrieval off.
- `web_api/` — request limits, the visitor address, response shapes, health, the app without retrieval.
- `service_settings/` — shipped defaults, environment overrides and bad values.
- `import_boundaries/` — which package may import which (checked in fresh interpreter processes).
