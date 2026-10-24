# tests/golden_sets — AGENTS.md
Purpose: tests of the question sets.

## Files
- `test_held_out_passage_questions.py` — fixture questions fg-3 and fg-4 are located in part 1 with a hit inside their gold sentence, fg-1 and fg-2 are dropped, and the block split gives development, test and edge-dropped results on synthetic centres.
- `test_fact_dev_question_set.py` — about 40 questions across all five books with evidence fields, none close to the 30, pair duplicates and missing evidence are caught.
- `test_other_question_sets.py` — fact-set flags, the unanswerable and regression sets, and the JSON round trip.
