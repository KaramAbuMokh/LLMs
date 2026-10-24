# heading_reading — AGENTS.md
Purpose: reads the chapter headings that are still readable in the RAW book files (the cleaned part texts lost their line breaks). `source_alignment/chapter_alignment.py` then searches these expected headings in the cleaned text. Pure text work: no model, no network, no book text in any message.

## Files
- `raw_chapter_headings.py` — the raw book file names by book number and the public reader of one raw book's headings: contents list when the file has one, else the body.
- `contents_list_reading.py` — reads the contents list at the top of a raw book (Storm, Game, Feast), stops before the appendix, splits names glued together ("CATELYNJON", "EddardTyrion").
- `body_heading_scanning.py` — reads headings from the body of a book without a contents list: whole-capital lines and "prose. NAME" tails (Dance), or "CHAPTER <NUMBER> NAME" lines with lost numbers and names filled in (Clash).
