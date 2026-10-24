# mixed_run

Purpose: the books plus question-answer run of the Game of Thrones fine-tune (one shuffled stream of question-answer pairs and book passages, started from the original 2025 model). Only the Kaggle-side code is kept in this repository; its Mac-side dataset builder, staging data and results stay private.

## Folders
- `mixed_kernel_source/`: the modules that run on Kaggle (entry point, examples, settings, evaluation, training, the small sanity ladder) and `fact_questions.py`, the 30 fact questions with the word rules that mark an answer right, which the evaluation harness reuses.
