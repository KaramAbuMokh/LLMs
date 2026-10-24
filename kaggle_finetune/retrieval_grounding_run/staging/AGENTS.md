# staging — AGENTS.md
Purpose: the payload of the PRIVATE Kaggle dataset `karamabumokh/got-gpt2-grounded-examples`, exactly as uploaded. Rebuilt by `../build_grounding_dataset.py`; never hand-edit. The sequences contain book text as token ids (copyrighted: private only, like the other Kaggle datasets).

## Subfolders
- `grounding_data/` — `grounded_train.json` (training sequences), `grounded_oracle_validation.json` (held-out pairs with their answer passage), `grounded_retrieved_validation.json` (development questions with the passages the Space retrieves), `build_report.json` (counts), `dataset-metadata.json` (Kaggle metadata).
