# kaggle_finetune

Purpose: the training code that runs on a free Kaggle T4 GPU: the shared fine-tuning kernel, the books plus question-answer run, and the grounding run that taught the model to answer from a passage placed in front of the question. Code only: training data, run results, generated notebook folders and weights stay in Karam's private working folder and are never committed.

## Folders
- `kernel_source/`: the shared modules (GPT-2 network, evaluation, example building, training loop, run settings) that every run reuses.
- `mixed_run/`: the code of the books plus question-answer run (only its `mixed_kernel_source/` is kept here; it holds the 30 fact questions and their word rules that the evaluation reuses).
- `retrieval_grounding_run/`: the grounding fine-tune of the retrieval work: the Mac-side dataset builder, the Kaggle-side training code, the bundle builder and the push script.

## Rules
- The grounding run's bundle builder packs `kernel_source/` and `mixed_run/mixed_kernel_source/` together with its own `grounding_kernel_source/` into one private Kaggle script.
- Paths such as `kaggle_finetune/mixed_run/staging/...` and `mixed_run_results/model.pth` name private data folders that exist only on Karam's Mac.
