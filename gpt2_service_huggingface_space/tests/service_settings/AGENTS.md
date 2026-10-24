# tests/service_settings — AGENTS.md
Purpose: tests of `gpt2_demo_service/service_settings.py` and its environment readers.

## Files
- `test_service_settings.py` — an empty environment gives exactly the shipped defaults, documented default numbers, each of the nine overrides parses, the pair search needs both its switch and a pair layout (and a pair layout without the switch stops the start), the server defaults of the request fields, every bad value raises `ValueError` naming its variable, empty values keep defaults, the Hub token is never stored.
