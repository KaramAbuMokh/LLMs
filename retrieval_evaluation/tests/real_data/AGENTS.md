# tests/real_data — AGENTS.md
Purpose: the check that needs the real books and pairs (marker `data`, skipped by default; run with `python -m pytest -m data`). Prints no book text.

## Files
- `test_real_data.py` — builds the golden sets on the real data and expects 816 answerable, 443 located, 189 development, 196 test and the three word-for-word pair questions; the build itself fails if a fact-development check finds a problem.
