# gpt2_demo_service — AGENTS.md
Purpose: the code that runs inside the Hugging Face Space: the two GPT-2 models, retrieval over the Game of Thrones books, grounded answers with
sources, and the web API. One package per job; the design and every signature are in `../../docs/DESIGN.md` (sections 5 to 7).
Dependency direction (checked by `../tests/import_boundaries/`): `language_models` and `index_loading` import no other service package;
`index_search` uses `index_loading` and the stage timer only (no torch, no pydantic, no fastapi); `grounded_answers` uses `index_search`,
`language_models` and the stage timer; `monitoring` stands alone; `service_startup` starts retrieval and may import the lower packages but not `web_api` or `service_settings`; `web_api` wires everything. All `__init__.py` files are empty package markers.

## Files
- `service_settings.py` — EVERY default value of the service in one place (index source, models and the 25-second answer time limit, search numbers, prompt numbers, public limits), the `ServiceSettings` bundle, `load_service_settings(environment)` for the ten documented environment overrides, the default values of the request fields a caller may leave out (answer length and the four sampling numbers) and `read_hub_token`. Only `web_api/app_factory.py` imports it.
- `settings_from_environment.py` — typed readers (text, true/false, positive whole number, choice, full commit hash) that take a mapping and name the variable in every error.

## Subfolders
- `language_models/` — the GPT-2 network with a preallocated key/value cache, next-token sampling, the Game of Thrones text writer (stop rules and time limit), the spam classifier and the weights download.
- `index_loading/` — downloads the private retrieval bundle at a pinned commit and loads it only after checking checksums, versions, encoder fingerprint and counts.
- `index_search/` — lexical, dense and fused search, the evidence gate and the pair examples; returns scored passages and pairs.
- `grounded_answers/` — prompt layouts, token-budgeted packing, public citations (one 25-word book snippet per answer), the copy guard with its index of every 26-word run of the books, and the one answer path the service and the evaluation share.
- `service_startup/` — starts retrieval at boot: encoders, bundle check, retriever, the copy guard's index and the status `/health` shows.
- `monitoring/` — memory numbers, the per-stage stopwatch and the one-line JSON log per answer (no visitor or book text).
- `web_api/` — the FastAPI app: request and response shapes, routes, public limits, the visitor address, the try-it page.
