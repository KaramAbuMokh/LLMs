# source_alignment — AGENTS.md
Purpose: finds where things sit in the books. It gives the passage cutter its chapters and gives every question-answer pair its book and chapter (DESIGN.md 8.3 and 8.5).
Chapters come from headings that are still readable in the raw book files and are searched, in reading order, in the cleaned part texts (where the line breaks are gone and
letters are often split: "C ATELYN", "PROLOGU E", "BRIENNE I t was"). Pairs are placed by the frozen rare-shingle locator of the contract.

## Files
- `book_part_texts.py` — loads the seven cleaned part texts (six training parts and the held-out stretch of A Feast for Crows) and checks them against the contract's part table.
- `heading_patterns.py` — turns an expected heading into a pattern for the cleaned text (one optional space between letters, a capital after it, no longer capitalised word around it) and formats chapter titles.
- `chapter_alignment.py` — takes the expected headings (from `heading_reading/`) and finds every one going forward, cuts each part into chapter spans that start after the heading, carries a chapter across a part edge (the three Feast parts are one sequence), cuts the last chapter at the appendix, applies the 95% gate per book and reports expected / found / unmatched / shortest / longest.
- `pair_placement.py` — the five placement rules for one pair: shingles inside its coarse window, shingles anywhere, window inside one chapter, window only, unplaced.
- `pair_records.py` — reads the 34,001 pairs, builds the shingle index once, places every pair, returns the stored records and the counts by method.

## Subfolders
- `heading_reading/` — reads the expected chapter headings from the raw book files: the raw file names and public reader, the contents-list reader and the body-heading scanner.
