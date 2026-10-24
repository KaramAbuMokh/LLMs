# passage_chunking — AGENTS.md
Purpose: cuts the book parts into sentence-aligned passages of about 200 GPT-2 tokens that never cross a chapter edge (DESIGN.md 8.4, D2). Text between chapters that belongs to no chapter
(the appendix after the last chapter, or a whole book that failed the alignment gate) is cut the same way and ships without a chapter number; the text after the last chapter of a part is titled "Appendix" (chunker version 2, so a citation says "Appendix" instead of showing no chapter; the live index, built with version 1, still shows none until the next rebuild), any other chapterless stretch carries no title; the bare headings between chapters are not indexed.

## Files
- `gpt2_token_counting.py` — counts GPT-2 tokens of a text (the unit of passage sizes and of the model's 1,024-token window).
- `sentence_spans.py` — finds sentence boundaries in the cleaned books (end mark, closing quotes, spaces, then a capital or opening quote) and cuts a sentence longer than a passage at word boundaries.
- `passage_boundary_planning.py` — decides which sentences form each passage: add sentences up to the target, never beyond the hard ceiling (checked on the real joined text), repeat the last sentence of the previous passage when it is short enough, merge a tiny last passage into the one before.
- `chapter_chunking.py` — turns the plan into stored passage records (id, book, chapter, character span, token count, exact text), one chapter or chapterless stretch at a time, for all parts. Holds the chunker version stored in the manifest and the "Appendix" title.
