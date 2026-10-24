# tests — AGENTS.md
Purpose: unit tests of the grounding run that need no network, no GPU and no book text.

## Files
- `conftest.py` — points the tokenizer cache at the mixed dataset folder.
- `test_oracle_passage_finding.py` — the oracle rule: most hits wins, two hits at least, never the evaluation's part, ties keep the earlier passage.
- `test_grounded_prompt_building.py` — a grounded sequence ends with the trained template, one end token, answer cut to the maximum length.
- `test_grounding_stream.py` — the 70/15/15 shares, hidden prompt labels, padding.
