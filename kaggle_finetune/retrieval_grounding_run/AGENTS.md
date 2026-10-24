# kaggle_finetune/retrieval_grounding_run — AGENTS.md
Purpose: Phase B of the retrieval work (workflow `wf-1024-2000-got-rag`, DESIGN.md section 12, DECISIONS `hq.gpt2_got_rag_1024`): a free Kaggle GPU fine-tune that teaches the live Game of Thrones model to USE a retrieved book passage. The model starts from the books + Q&A mixed weights that are live in the Space (checksum prefix 528ef774), trains one epoch on one shuffled stream (70 percent grounded sequences, 15 percent plain question-answer pairs, 15 percent book blocks, loss on the answer and end token only), and writes a NEW weights file; the live model is never overwritten. Everything on Kaggle is private; free T4 only; no OpenAI calls.

## Files
- `build_grounding_bundle.py` — packs the grounding modules and the shared modules of the earlier runs into one Kaggle script for a mode (`grounding_smoke`, `grounding_full`, `dry_grounding`) and writes the private notebook metadata.
- `push_grounding_run.sh` — builds the bundle and pushes it (that starts the run on a free T4), then shows the status. If the CLI prints an error, check the status before pushing again.
- `pytest.ini` — test paths and import paths for the Mac modules and the kernel modules.

## Subfolders
- `grounding_dataset_builder/` — the Mac side: finds each training pair's answer passage, packs grounded prompts with the Space's own packer and writes the dataset payload.
- `grounding_kernel_source/` — the training code that runs on Kaggle.
- `staging/` — the dataset payload exactly as uploaded (`karamabumokh/got-gpt2-grounded-examples`, private).
- `kernel_push/` — generated upload folder for the notebook.
- `tests/` — unit tests of the oracle rule, the prompt packing and the training stream.
Run the tests with `python -m pytest` in this folder. Results of a run land in `../retrieval_grounding_results/`.
