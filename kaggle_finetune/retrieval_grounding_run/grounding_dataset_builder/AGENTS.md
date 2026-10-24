# grounding_dataset_builder — AGENTS.md
Purpose: the Mac side of the retrieval-grounding run (DESIGN.md section 12). It turns the 34,001 training pairs into grounded training sequences with the SAME retriever and prompt packer the Space uses, and writes the private Kaggle dataset payload to `../staging/grounding_data/`. Run from the project folder:
`TIKTOKEN_CACHE_DIR=<mixed dataset folder> PYTHONPATH=gpt2_service_huggingface_space:. python kaggle_finetune/retrieval_grounding_run/grounding_dataset_builder/build_grounding_dataset.py` (service venv; takes about 25 minutes).

## Files
- `grounding_data_settings.py` — every number and path: oracle share 80 percent, 2 or 3 hard distractors, 300 validation pairs, the Space's token budget, the seed.
- `oracle_passage_finding.py` — finds the passage that holds a pair's answer (at least two of the pair's rare shingle hits inside it; never the evaluation's own part).
- `grounding_prompt_building.py` — packs a grounded prompt with the service's own packer and turns it plus the answer into a training sequence (answer cut to 142 tokens, one end token).
- `grounding_example_building.py` — builds the sequence of one pair (oracle plus distractors, or distractors only, shuffled) and the retrieved validation sequences.
- `build_grounding_dataset.py` — the command: loops over the pairs with the real retriever, splits off the validation slice, writes the files and a build report (19,289 training sequences, 300 oracle validation, 189 retrieved validation on 2026-10-24).
