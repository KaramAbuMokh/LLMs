# grounding_data — AGENTS.md
Purpose: the exact payload of the PRIVATE Kaggle dataset `karamabumokh/got-gpt2-grounded-examples`, the training and validation sequences of the grounding fine-tune. Written by `../../grounding_dataset_builder/build_grounding_dataset.py`; never hand-edit. The sequences hold book text as GPT-2 token ids (copyrighted): private only, never uploaded to the Space or any public place.

## Files
- `grounded_train.json` — the training sequences (19,289: 15,450 with the pair's answer passage in front of the question, the rest plain), about 430 tokens each on average.
- `grounded_oracle_validation.json` — 300 held-out pairs, each with its answer passage ("oracle" evidence).
- `grounded_retrieved_validation.json` — 189 development questions with the passages the Space's own search retrieves.
- `build_report.json` — the counts above plus the 14,412 pairs that found no answer passage.
- `dataset-metadata.json` — the Kaggle title, id and license line used when the dataset is pushed.
