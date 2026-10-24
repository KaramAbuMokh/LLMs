# metrics — AGENTS.md
Purpose: pure scoring functions (DESIGN.md section 11.2). No model and no network; the only file access is loading the Kaggle fact rules.

## Files
- `ranking_metrics.py` — hit, precision, recall, reciprocal rank and normalized discounted cumulative gain over relevance flags, hit and precision within a packed token budget (the only measure used to compare chunk sizes), mean and percentile helpers.
- `passage_relevance.py` — a passage is relevant to a held-out question when it contains one of the question's gold hits whole; counts relevant passages in a whole index.
- `answer_overlap_scoring.py` — token F1, the comparable form of a question text, key-term recall of the gold answer (terms the question already has do not count), and the detector for capitalised names that appear in neither the question nor the packed context.
- `copyright_checks.py` — the A16 check: a response may not share more than 25 consecutive words with any passage of the bundle, may carry at most one book snippet, and each snippet is at most 25 words. Written independently of the service's runtime guard.
- `fact_rule_loading.py` — loads the 30 fact questions and their word rules from the Kaggle source together with the source's own judging function, so answers are scored exactly as the 10 of 30 baseline was.
- `answer_run_scoring.py` — the record of one generated answer (ids, scores, counts, the model's text, no passage text) and the scores of a configuration: the fact majority and the held-out means.
