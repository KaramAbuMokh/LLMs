# tests/import_boundaries — AGENTS.md
Purpose: keeps the package structure honest.

## Files
- `test_import_boundaries.py` — in fresh interpreter processes: importing the retriever loads no torch, pydantic or fastapi, and the answer path loads no web stack; by reading the sources: only `web_api/app_factory.py` imports `service_settings`, every `__init__.py` is empty, lower packages never import upper ones.
