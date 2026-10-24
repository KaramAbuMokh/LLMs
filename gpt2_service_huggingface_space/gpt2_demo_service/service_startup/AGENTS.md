# service_startup — AGENTS.md
Purpose: starts the retrieval part of the service when the Space boots. It is wiring, so it may import every lower package (`index_loading`, `index_search`, `grounded_answers`) but never `web_api` or `service_settings` (checked by `../../tests/import_boundaries/`). Any failure turns retrieval OFF with a reason and the service still starts.

## Files
- `retrieval_startup.py` — loads the dense encoder and the optional reranker (`load_encoders`), validates the bundle, builds the retriever, builds the copy guard's index of book runs (`build_book_run_index`) and runs one warm-up search (`assemble_retrieval`); returns the retriever, the book-run index and the `RetrievalStatus` that `/health` shows (ready, index version and short revision, passage, pair and book-run counts, encoder, reranker, layout, problem).
