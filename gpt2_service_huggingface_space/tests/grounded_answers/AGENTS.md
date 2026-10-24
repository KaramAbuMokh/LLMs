# tests/grounded_answers — AGENTS.md
Purpose: tests of `gpt2_demo_service/grounded_answers/` with invented text only (no book text).

## Files
- `conftest.py` — builders for invented passages, pairs and evidence of any size.
- `test_prompt_layouts.py` — byte-exact prompt text: question-only equals the old service's prompt, best item last, ignored item kinds.
- `test_context_packing.py` — 150 random property trials (never over 1,024 with the reserve, question intact and last, budget kept, only the last passage shorter than its record), trimming, the 40-token rule, overlap skipping with adjacent chunks allowed, layout kinds, zero budget.
- `test_source_citations.py` — exactly one book snippet on the first passage, same snippet for different questions, at most 25 words, pair snippet is its question, snippet switch, chapter labels.
- `test_book_run_index.py` — a run of 26 words is found and 25 are not, unknown words never match, neighbours sharing a sentence read as one stretch (a run may cross the passage edge), a gap or another part breaks the stretch, deterministic count, empty index.
- `test_verbatim_guard.py` — the 33-word Ysolde fixture passage is cut at a sentence end, the no-sentence ellipsis cut, exactly 25 words allowed, a copy of a passage that was never packed is cut too, separate passages never join, case and apostrophes folded, no index means no check.
- `test_grounded_answer_service.py` — `answer_question` equals `answer_from_evidence`, routes (off, no evidence, failed, from the packed prompt), sources, answer reserve clamping, guard and time limit, stage timings, memorised book text cut on every route, a question that cannot fit refused before the search, the reserve shrunk at the limit.
