# tests/source_alignment — AGENTS.md
Purpose: tests of `source_alignment/` on invented text.

## Files
- `test_raw_chapter_headings.py` — contents lists with glued names, Dance-style glued tails, Clash-style numbered chapters with lost lines, number spelling, file-level reading.
- `test_heading_patterns.py` — split-letter forms ("C ATELYN", "PROLOGU E", "BRIENNE I t was", "CHAPTER ONE ARYA"), page numbers, look-alikes that must not match, title formatting.
- `test_chapter_alignment.py` — forward matching, a lost heading, the appendix cut, a chapter crossing a part edge, the 95% gate, stray names inside the minimum chapter distance, chapter lookup edges.
- `test_pair_placement.py` — each of the five placement methods, window clamping, a span centre next to a chapter edge, never placed outside the window.
