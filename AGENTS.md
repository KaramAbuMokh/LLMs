# LLMs

Purpose: the code behind Karam Abu Mokh's Game of Thrones GPT-2 and its live demo (Hugging Face Space karamAM/gpt2-demos, shown on karamam.com): a GPT-2 medium that searches the five books before it answers and returns its sources, plus the offline index builder, the evaluation harness and the Kaggle training code. A public repository. Read the `AGENTS.md` of any folder before editing it.

## Rules
- The books are copyrighted: no book text, no retrieval index, no weights, no generated results and no training data are ever committed (`.gitignore` keeps them out). Answers show at most one 25-word snippet.
- Never commit a token or a key. Scripts read a Hugging Face token from the environment or from the file named in `HF_TOKEN_FILE`.
- Files stay under 150 lines, every statement carries a short comment, names are spelled out in full.
- Run the tests of a package from inside its folder with `python -m pytest`.

## Files
- `README.md`: what the retrieval system is, how a question travels through it, the before and after numbers, the targets that were missed, how to run and test it.
- `.gitignore`: keeps weights, caches, generated bundles, results and training payloads out of git.
- `AGENTS.md`: this file.

## Folders
- `gpt2_service_huggingface_space/`: the service that runs on the Space (Docker, FastAPI, seven packages, tests, the frozen contract and the shared encoders).
- `retrieval_index_builder/`: offline, builds the private index of passages, pairs, vectors and word postings.
- `retrieval_evaluation/`: offline, question sets, metrics, ablations, calibration, answer runs and the ship-gate decisions.
- `retrieval_integration/`: scripts that start the real retrieval path, score candidate models and verify the live Space.
- `kaggle_finetune/`: the Kaggle training code (shared kernel, books plus question-answer run, grounding run).
- `docs/`: the design document and the results write-up.
- `gpt2_service/`: the first version of the service, before retrieval (kept for reference).
