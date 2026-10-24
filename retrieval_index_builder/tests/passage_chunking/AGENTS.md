# tests/passage_chunking — AGENTS.md
Purpose: tests of `passage_chunking/` on invented chapters.

## Files
- `test_sentence_spans.py` — sentence boundaries with quotes and speech tags, exact offsets, cutting a sentence longer than a passage.
- `test_chapter_chunking.py` — (the appendix tail is titled "Appendix") exact substrings, ceilings for 128/150, 200/230 and 256/290, one-sentence overlap and when it is skipped, tiny-tail merge, no passage across a chapter edge, chapterless text, unique ids.
