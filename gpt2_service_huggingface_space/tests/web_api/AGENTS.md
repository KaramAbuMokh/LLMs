# tests/web_api — AGENTS.md
Purpose: tests of `gpt2_demo_service/web_api/` through an in-process client; the app is built from fakes (scripted model, fixture bundle), so no weights are needed.

## Files
- `conftest.py` — the app builder (`make_client`), a fake spam model and a ready status.
- `test_generate_route.py` — the old portfolio body, exact key sets of both bodies and of every source (including a source with no book or chapter), spam body exactly `{"completion"}`, `use_retrieval: false`, `Server-Timing`, 422/429/503 shapes, lowered limits, a spoofed forwarded-for header cannot reset the visitor limit, an emoji question gets 422 and gives its place back, empty stop texts and long task names are refused, omitted fields use the server defaults, the log holds no caller text, place in line returned after a model error, OpenAPI shows both shapes.
- `test_health_and_page.py` — old `/health` keys kept, the retrieval block, the reason when retrieval is off, the try-it page.
- `test_request_limits.py` — line cap, per-visitor window and expiry, counter never below zero, idle visitors forgotten, refused callers never stored.
- `test_visitor_address.py` — the proxy chain read from the right: public entries win, private and loopback hops are skipped, IPv6, malformed entries, no header.
- `test_app_without_retrieval.py` — kill switch, unpinned revision and a broken bundle keep `/generate` answering with the plain prompt.
