# grounded_answers — AGENTS.md
Purpose: turns a question and retrieved evidence into an answer with sources, in the model's exact trained template. Needs torch and tiktoken
(through `language_models`), no fastapi, no pydantic. The service and the evaluation both call `answer_from_evidence`, so what is measured is what ships.

## Files
- `answer_settings.py` — the four prompt layouts and the answer settings (layout, retrieved-token budget, safety margin, answer floor and ceiling, snippet switch), without defaults.
- `prompt_layouts.py` — the byte-exact training template ("qusetion" typo included) and the four ways retrieved text sits in front of it; the best item is rendered last, nearest the question.
- `context_packing.py` — fits examples and passages into the token budget: best first and whole, only the last passage trimmed at a sentence end (kept at 40 tokens or more), heavy overlaps skipped, then the real token count re-checked.
- `source_citations.py` — builds the public sources: book and chapter for each, ONE book snippet of at most 25 words (the opening of the first passage, independent of the question), a pair's snippet is its question.
- `book_run_index.py` — the copy guard's memory: one 64-bit hash per run of 26 consecutive words across ALL passages of the bundle (neighbours that share a sentence are joined first, so a run may cross a passage edge), kept as a sorted array of about 14 MB, built at start-up in about 2 seconds; same word rule as the evaluation's independent check.
- `verbatim_guard.py` — cuts an answer at the first stretch that repeats more than 25 consecutive words of ANY book passage (every route, packed or not), back to the last sentence end; without an index (retrieval off) nothing is checked.
- `grounded_answer_service.py` — `answer_question` (refuses a question that cannot fit with `QuestionTooLongError` before any search, then searches and follows the shared path; falls back to the plain prompt when retrieval is off or fails) and `answer_from_evidence` (pack, generate, guard, cite, name the route; the answer reserve never exceeds the room left in the context).
