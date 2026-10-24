# tests/metrics — AGENTS.md
Purpose: tests of the pure scoring functions.

## Files
- `test_ranking_metrics.py` — hand cases (hit at 3, precision, recall, reciprocal rank, normalized discounted cumulative gain 0.650921 at 5 and 0.386853 at 3, hit within the packed budget).
- `test_passage_relevance.py` — a passage holding a hit whole is relevant, one cutting it is not.
- `test_answer_overlap_scoring.py` — token F1, key-term recall, capitalised names and the invented-name detector.
- `test_copyright_checks.py` — a 26-word copy of an unpacked passage and a response with two snippets are flagged (invented text only).
- `test_fact_rule_loading.py` — 30 rules whose first ten are the known ten, the source's own judge, no module left registered.
- `test_answer_run_scoring.py` — the two-of-three majority rule and the held-out means.
