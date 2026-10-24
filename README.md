# Game of Thrones GPT-2 with a book search (retrieval-augmented generation)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)

A GPT-2 medium that I fine-tuned on the five *A Song of Ice and Fire* books now **searches the books before it answers**, reads the best passages, and returns its sources with the answer. It runs on a free CPU Hugging Face Space and is the live demo on my portfolio.

- **Try it:** [huggingface.co/spaces/karamAM/gpt2-demos](https://huggingface.co/spaces/karamAM/gpt2-demos) (the first question after a quiet spell wakes the Space, about a minute)
- **Case study:** the GPT-2 page on [karamam.com](https://karamam.com)
- **What this branch adds:** the whole retrieval system, split into a runtime service, an offline index builder, an evaluation harness and a grounding fine-tune, with 477 tests. The first version of the service (before retrieval) is kept in [`gpt2_service/`](gpt2_service/README.md).

## The problem

The model sounded like the books and guessed the facts. It once answered that Gregor Clegane sits on the Iron Throne. A model this small cannot memorise five long novels, so the fix is to let it read the right page before it answers, which is what retrieval does.

## What happens when you ask a question

```
OFFLINE, once on a laptop                         retrieval_index_builder/
  five books (text) ---> chapter alignment ---> 128-token passages, each with book + chapter     26,112
  34,001 question-answer pairs ---> placed in the books ---> tagged with book + chapter
  passages + pairs ---> dense vectors (bge-small, ONNX) + word index (BM25) ---> manifest with checksums
  ---> uploaded to a PRIVATE Hugging Face dataset repo, pinned by commit

TRAINING, once, free Kaggle T4, 68 minutes        kaggle_finetune/retrieval_grounding_run/
  19,289 questions that come with their passage ---> the books+Q&A model learns to answer FROM the passage

RUNTIME, Docker Space, 2 CPU, 16 GB               gpt2_service_huggingface_space/
  start:   weights + index downloaded with a secret ---> checksums and encoder fingerprint verified ---> one warm-up query
           (any retrieval failure switches retrieval off with a reason; the service still starts)
  request: POST /generate { prompt, task_name: "got" }
    question ---> word search (top 50) + meaning search (top 50) ---> reciprocal rank fusion
             ---> re-rank the top 10 with a cross-encoder ---> keep evidence above a calibrated score floor
             ---> pack up to 3 passages (at most 600 tokens) in front of the question, in the exact trained template
             ---> GPT-2 medium with a preallocated key/value cache, 25 second limit
             ---> copy guard: cut any stretch of more than 25 words shared with any passage of the books
             ---> { "completion", "retrieval_used", "sources": [ { book, chapter, snippet of at most 25 words } ] }
```

Pasting passages in front of the old model did **not** help (it ignored them: 12 facts right fell to 10), so the retrieval fine-tune on Kaggle is part of the system, not an extra.

## Architecture rules

```
retrieval_contracts   frozen, standard library only: record types, bundle layout, text rules, encoder interface
retrieval_encoders    numpy + onnxruntime + tokenizers: the dense encoder and the re-ranker, shared by builder, service, evaluation
gpt2_demo_service/
  language_models     torch + tiktoken: the GPT-2 network, KV cache, sampling, spam classifier. Imports no retrieval code
  index_loading       downloads and validates the bundle. No torch, no pydantic
  index_search        word, meaning and fused search, the evidence gate. No FastAPI, no torch
  grounded_answers    prompt layouts, token-budget packing, citations, copy guard, the ONE answer path
  service_startup     boots retrieval and reports its state
  monitoring          memory, stage stopwatch, one JSON log line per answer (no visitor or book text)
  web_api             FastAPI: the only package that imports it; wires everything together
  service_settings    every default value of the service in one file (ten environment overrides)
retrieval_index_builder   offline; imports the contract and encoders, never the service
retrieval_evaluation      offline; imports the contract, encoders and service packages, never the builder
```

Every import boundary above is a test (`gpt2_service_huggingface_space/tests/import_boundaries/`, `retrieval_evaluation/tests/test_package_rules.py`). The code also keeps files under 150 lines, a comment on every statement and full-word names, and every folder holds 5 to 8 files and its own `AGENTS.md` (the evaluation package tests these rules too).

## Results

Same questions, same seeds, same machine. "Right" means 2 of 3 samples contain the accepted words and none of the rejected ones.

| Measure | Live model alone (before) | Same model + passages pasted in | Grounded model + passages (now live) |
|---|---|---|---|
| 30 fact questions right | 12 | 10 | **17** |
| 42 new fact questions (never trained or tuned on) | 2 | 8 | **13** |
| The four old mistakes (Iron Throne twice, Kingslayer, Theon's home) | 1 of 4 | 1 of 4 | **4 of 4** |
| Answers that name a person found nowhere in the question or passages (196 held-out questions) | 36% | 28% | **12%** |
| Key words of the reference answer found (same 196 questions) | 0.052 | 0.078 | **0.213** |

Finding the right passage (196 test questions, plus or minus 6 points): in the top 5, 0.714 with words and meaning search, **0.781** with the re-ranker added. Passage size was chosen by measurement: the right passage fits the 600-token budget for 66% of questions with 128-token passages, 63% with 200, 57% with 256.

Live on the Space (no token, as a visitor): 19 of 30 facts right with one try each, 0 failed calls, p50 3.3 s and p95 4.3 s per answer (the free CPU varies about 2x between hours, p95 up to 10.1 s), memory 3.5 GB of 16, and 0 copyright findings against all 26,112 passages.

**Targets that were not met**, stated plainly: the facts count missed its target of 20 (17 reached, the gate of 15 passed); 3 of the 12 old right answers turned wrong (limit 2); the first-ten questions scored 5 where the gate was 6; 35 to 38% of 40 unanswerable questions still get passages (limit 15%), because half of them sound like the series and a search cannot tell; and no stored question-answer example reached the precision bar, so none is ever shown to the model (the code path stays behind an explicit switch). The full write-up is [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository map

| Folder | What it is |
|---|---|
| [`gpt2_service_huggingface_space/`](gpt2_service_huggingface_space/AGENTS.md) | The service that runs on the Space: Dockerfile, the seven packages above, tests, the frozen contract and the shared encoders |
| [`retrieval_index_builder/`](retrieval_index_builder/AGENTS.md) | Offline: chapter alignment, passage chunking, pair placement, vectors, word index, manifest, upload |
| [`retrieval_evaluation/`](retrieval_evaluation/AGENTS.md) | Offline: question sets, metrics, retrieval ablation, calibration, answer runs, decision rules, live checks |
| [`retrieval_integration/`](retrieval_integration/AGENTS.md) | Scripts that start the real retrieval path locally, score a candidate model, compare temperatures and verify the live Space |
| [`kaggle_finetune/`](kaggle_finetune/AGENTS.md) | Kaggle training code: the shared kernel, the books+Q&A run, and the grounding run that taught the model to use passages |
| [`docs/`](docs/AGENTS.md) | `DESIGN.md` (goals, 24 decisions with rejected options, data flow, module signatures) and `RESULTS.md` |
| [`gpt2_service/`](gpt2_service/README.md) | The first version of the service, before retrieval (DigitalOcean, one model in memory at a time) |

## API

- `GET /health`: models loaded, memory, and the retrieval status (index version, passage count, whether the re-ranker is on, the reason if retrieval is off).
- `POST /generate` with `{"prompt": "...", "task_name": "got" | "spam", "max_new_tokens": 120}`; optional `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop`, `use_retrieval`.
  - `got` returns `{"completion", "retrieval_used", "sources": [{"source_type", "book", "chapter", "snippet"}]}`. `completion` is the old contract and is unchanged, so existing clients keep working. A `Server-Timing` header carries the time spent per stage.
  - `spam` returns exactly `{"completion": "spam"}` or `{"completion": "not spam"}`.
- Errors: `422` bad input (including a question over 864 tokens), `429` too many requests (20 per minute per visitor), `503` line full (`Retry-After` set).

## Run it

Python 3.11. The model weights, the book text and the retrieval index are **not in this repository**: the books are copyrighted and the index and weights live in private Hugging Face repos.

```bash
# tests (no weights, no network needed for the unit tests)
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r gpt2_service_huggingface_space/requirements.txt -r gpt2_service_huggingface_space/requirements-test.txt \
  -r retrieval_index_builder/requirements.txt
(cd gpt2_service_huggingface_space && python -m pytest)            # 313 tests, including the 11 contract self-tests
(cd retrieval_index_builder && python -m pytest)                   # 78 tests
(cd retrieval_evaluation && python -m pytest)                      # 77 tests
(cd kaggle_finetune/retrieval_grounding_run && python -m pytest)   # 9 tests
```

```bash
# the service: the Docker image is the supported way (it bakes in the two public encoder models)
docker build -t gpt2-demos gpt2_service_huggingface_space
docker run -p 7860:7860 -e HF_TOKEN=<read token> -e WEIGHTS_REPO_ID=<you>/<your weights repo> gpt2-demos
# without Docker and without retrieval (plain prompt), from gpt2_service_huggingface_space/
GOT_RETRIEVAL_ENABLED=false HF_TOKEN=<read token> WEIGHTS_REPO_ID=<you>/<your weights repo> \
  uvicorn gpt2_demo_service.web_api.app_factory:create_production_app --factory --port 7860
```

The index repo id and its pinned commit are in `service_settings.py`; point them at your own private index repo.

Quick switches (Space variables, restart about 35 s): `GOT_RETRIEVAL_ENABLED=false` (plain prompt), `GOT_RETRIEVAL_RERANKER_ENABLED=false`, `GOT_WEIGHTS_FILE_NAME=<older model file>`. The whole list is in `service_settings.py`.

```bash
# offline pieces, from the repository root (the data folders they read are listed in each AGENTS.md)
export HF_TOKEN_FILE=/path/to/file-with-a-hugging-face-token       # only the upload steps read it
PYTHONPATH=gpt2_service_huggingface_space:. python -m retrieval_index_builder.build_index_command [--upload]
PYTHONPATH=gpt2_service_huggingface_space:. python -m retrieval_evaluation.run_evaluation_command golden|retrieval|calibrate|answers|report|live|all
PYTHONPATH=gpt2_service_huggingface_space:. python retrieval_integration/live_space_verification.py   # token-free check of the deployed Space
bash kaggle_finetune/retrieval_grounding_run/push_grounding_run.sh                                    # builds the Kaggle script and starts the grounding run
```

## Decisions worth knowing

- **One answer path.** The service and the evaluation call the same `answer_from_evidence`, so the numbers above measure the code that ships.
- **Settings in one place, no hidden defaults.** Every value is in `service_settings.py`; each package declares its own settings type with no defaults, so a package never decides a number on its own.
- **Calibrated, not guessed.** The evidence floor (0.102 on the re-rank score) and the line cap (5 requests, from the slowest measured p95) come from measurements; the evaluation writes its ship-gate decisions with the evidence.
- **Failing closed.** A retrieval problem never becomes an error: the answer falls back to the plain prompt and `retrieval_used` is `false`. A bundle whose checksum, version or encoder fingerprint does not match is refused.
- **Copyright by construction.** The index is private; answers show at most one 25-word snippet; every answer on every route is checked against a hash of every 26-word run of all passages and cut at the first hit (3,000 random real stretches: every one of 26 or more words cut, none of 25 or fewer).
- **Found by review, fixed, and tested.** A senior review found real problems in the first release; the four worth naming: the per-visitor limit could be bypassed with a spoofed `X-Forwarded-For` header (the limit now keys on the rightmost public address the proxy wrote), a long emoji question returned 500 (now 422), the log carried a caller's task name, and the copy guard saw only the passages in the prompt.

## Limits of this version

The model is a 355M-parameter GPT-2 and still answers some questions wrongly: 17 of 30 fact questions right is better, not solved. It is a demo of the method (index, search, re-rank, grounded fine-tune, evaluation), not a general question-answering system. The live index was built with chunker version 1, so 650 appendix passages (2.5 percent) still cite no chapter until the next rebuild.
