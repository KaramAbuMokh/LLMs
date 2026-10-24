# GPT-2 service: a small API for my two fine-tuned GPT-2 models

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![DigitalOcean](https://img.shields.io/badge/DigitalOcean-0080FF?logo=digitalocean&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)

A FastAPI service that puts my two fine-tuned GPT-2 models behind one HTTP endpoint, packaged in Docker and deployed with
a small script. One model answers questions about *A Song of Ice and Fire* (Game of Thrones), the other classifies a text
message as spam or not spam. This is the serving side of the project; the training side is in two notebooks.

**Try the models live:** [huggingface.co/spaces/karamAM/gpt2-demos](https://huggingface.co/spaces/karamAM/gpt2-demos).
That Space is the newer version of this service (same request shape, see "Live demo and this repo" below). I first ran this
code in Docker on a DigitalOcean droplet; that server is no longer running.

## How it works

- `POST /generate` takes a prompt and a `task_name`, and returns `{"completion": "..."}`.
- `task_name: "got"` (or any value other than `spam`) loads the Game of Thrones model, GPT-2 medium (355M), from `model.pth`.
- `task_name: "spam"` is meant for the spam classifier, GPT-2 small (124M) with a 2-way output layer, from `spam_model.pth`.
- Only one model is kept in memory at a time: asking for the other task unloads the first, so the service fits a small
  server.
- The GPT-2 network (`GPT.py`, `GPTspam.py`) is my own PyTorch code; text is turned into tokens with `tiktoken`.
- Every request and the memory use around it are logged to `logs/app.log`.
- Only `/generate` is reachable. Every other path answers 403, and the API docs pages are switched off.

## Run it

The weight files are not in this repository (they are large and git-ignored). Train them with the two notebooks
([Game of Thrones](https://github.com/KaramAbuMokh/LLM-and-transformer-structure-explained),
[spam](https://github.com/KaramAbuMokh/fine-tune-GPT2-on-spam-email-classification)) and save each with
`torch.save(model.state_dict(), ...)`.

Locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cp /path/to/your/model.pth .          # Game of Thrones weights (spam_model.pth goes here too)
bash start.sh                         # serves on http://localhost:8080
```

With Docker:

```bash
docker build -t gpt2-service .
docker run -p 8080:8080 \
  -v /path/to/model.pth:/app/model.pth \
  -v /path/to/spam_model.pth:/app/spam_model.pth \
  gpt2-service
```

Ask a question:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who is Jon Snow?", "task_name": "got"}'
```

Request fields: `prompt` (1 to 2,000 characters), `task_name`, `max_new_tokens` (1 to 64), `temperature` (0.1 to 2.0),
`top_p`, `top_k` and `stop`.

Settings (environment variables):

| Variable | Default | What it does |
|---|---|---|
| `CKPT_PATH` | `model.pth` | Game of Thrones weights file |
| `MODEL_NAME` | `gpt2` | Tokenizer name for `tiktoken` |
| `TORCH_THREADS` | `1` | CPU threads for PyTorch |
| `API_KEY` | empty | If set, requests must send it as the `x_api_key` query parameter |
| `LOG_LEVEL` | `INFO` | How much the service logs |

## Deploy to a server

`deploy.sh` builds a tagged image, starts it on a spare port, waits for it to answer, then swaps it in on port 80 and
removes old images (keeping the last three). Step-by-step guides are in `docs/`:
[a fresh DigitalOcean droplet](docs/droplet_deployment.md), [everyday Docker commands](docs/docker_management.md) and
[updating a running server](docs/updating_main_branch.md).

## Project structure

```
.
├── app.py              # FastAPI app: the /generate endpoint, CORS, the "only /generate" rule, optional key
├── model.py            # loads the right model per task, generates text, keeps one model in memory
├── GPT.py              # GPT-2 network used for text generation
├── GPTspam.py          # the same network for the spam classifier
├── logging_utils.py    # file and console logging, memory usage in the log
├── test_api.py         # small script that calls a running service
├── Dockerfile          # python 3.11 slim image, CPU-only PyTorch, runs uvicorn on port 8080
├── start.sh            # starts the service locally
├── deploy.sh           # build, check, swap and clean up on the server
├── requirements.txt    # pinned packages
└── docs/               # deployment and Docker guides
```

## Live demo and this repo

The demo Space started from this code and changed a few things on purpose: it builds the exact prompt the model was
trained on, uses the sampling settings in the request, stops at the end token and returns only the answer, really
classifies spam, and loads both models together. This repository keeps the original service as I first wrote it, so a few
behaviours differ from the Space:

- The Game of Thrones answer is generated with fixed sampling (temperature 1.2, top-k 3) and 50 new tokens, whatever the
  request says, and it includes the prompt in the text.
- The spam task returns a "not available" message here; the `classify_review` code is in `model.py`, commented out of the
  call.
- `deploy.sh` and `test_api.py` check a `/health` path, which the "only `/generate`" rule now blocks, so that check needs
  updating before the deploy script can finish.
