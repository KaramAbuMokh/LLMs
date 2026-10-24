# web_api — AGENTS.md
Purpose: the FastAPI app, the only package that uses FastAPI. Retrieval start-up lives in `../service_startup/`. `app_factory.py` is the only module that reads the settings; routes get their state
from the app (components, request gate, inference lock, limits), so tests build the app from fakes with `create_app`.

## Files
- `app_factory.py` — loads both models and starts retrieval at startup (`load_service_components`), builds the app with CORS, the try-it page and the two routes (`create_app`), and `create_production_app` for `uvicorn --factory` (the only caller of `load_service_settings(os.environ)` and `read_hub_token`).
- `api_schemas.py` — the request (task name up to 32 characters, stop texts never empty, sampling fields optional: an omitted field gets the server default), `RequestDefaults` (those server defaults, filled by `service_settings.py`), the source shape (always four keys), the spam body `{"completion"}` and the Game of Thrones body `{completion, retrieval_used, sources}`; no `exclude_none` anywhere.
- `generate_route.py` — `POST /generate`: limits, request gate, single inference lock, spam or Game of Thrones path (omitted request fields filled from the server defaults), a question too long for the model answered with 422 "Questions can be up to N tokens.", `Server-Timing` header on Game of Thrones answers, one log line per answer, and one memory line labelled only `after_got` or `after_spam` (no caller text ever reaches the log).
- `health_route.py` — `GET /health`: the old keys plus a `retrieval` block (ready, index version and short revision, counts, encoder, reranker, layout, problem).
- `request_limits.py` — the public-safety settings record and the gate: places in line (503) and requests per visitor per minute (429); a refused caller is never stored, so a flood of new addresses cannot grow the table.
- `visitor_address.py` — the caller's address: the rightmost entry of `X-Forwarded-For` that is not a private, shared, loopback or link-local address (every proxy appends, so only the left side can be written by the caller; documentation ranges count as public on purpose), else the socket address.
- `try_it_page.html` — the browser page at `/`: question box, examples, the answer with its time, and a "From the books:" list of each source's book and chapter with the one snippet.
