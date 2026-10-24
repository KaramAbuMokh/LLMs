# grounding_kernel_source — AGENTS.md
Purpose: the code of the retrieval-grounding run that runs inside the PRIVATE Kaggle notebook `karamabumokh/got-gpt2-retrieval-grounding-finetune`. Kaggle runs one file, so `../build_grounding_bundle.py` packs these modules together with the shared modules of `../../kernel_source/` and two modules of `../../mixed_run/mixed_kernel_source/` (`mixed_settings.py`, `mixed_examples.py`). Edit here, then rebuild; never edit the bundle.

## Files
- `grounding_settings.py` — the switches on top of the mixed run's settings: batch sizes, the 70/15/15 shares, the gentle learning rate (2e-5), one epoch, evaluation every 400 steps, patience 3, the checksum prefix of the only valid start weights, and the tiny dry-run values.
- `grounding_stream.py` — the one shuffled training stream (grounded, plain pairs, book blocks; one kind per step; grounded batches sorted by length inside groups) and the step count of an epoch.
- `grounding_evaluation.py` — the four validation numbers (answer loss with the Space's retrieved evidence, with the oracle passage, with no evidence, and the book voice) and the seconds per step by kind.
- `grounding_entry.py` — finds the attached data and the live weights (checksum checked), tokenises, trains, saves the best weights as `model.pth`, writes the history, the curves, `grounding_comparison.json` (before/after table, plain-path flag, stop check on the 10 fixed questions, GPU minutes).
