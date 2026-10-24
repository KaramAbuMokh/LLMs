# golden_sets — AGENTS.md
Purpose: the question sets the evaluation uses (DESIGN.md section 11.1). Sets used to choose settings (G1 development, fact development) and sets that are only reported (G1 test, the 30 facts, the regression set) never mix.

## Files
- `golden_question_records.py` — the record types of all sets, the golden-sets aggregate with its counts, and the conversion to and from the JSON stored in the results folder.
- `held_out_passage_questions.py` — loads the seven book parts and the 816 answerable held-out questions, finds where each is answered with the shared locator (question plus answer), and cuts the located ones into development and test by ten contiguous blocks, dropping those near a block edge.
- `fact_question_set.py` — the 30 fact questions from the Kaggle source, flagged for the known ten and for the three that are also stored pair questions.
- `fact_dev_question_set.py` — 42 new fact questions across all five books with their word rules and, for each, the book part and offset where the fact is stated (no book text), plus three checks: not one of the 30 or close to one, not a stored pair question, and an accepted term of every group really near the evidence offset. The file name follows the design.
- `no_answer_questions.py` — 40 questions the books cannot answer: 25 unrelated and 15 that only the television series answers.
- `regression_questions.py` — the four questions the old model got wrong (two about the Iron Throne), with the words that must not appear.
