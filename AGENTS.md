# LLMs

Purpose: a FastAPI and Docker service that serves Karam Abu Mokh's two fine-tuned GPT-2 models (Game of Thrones answers and spam classification) behind one endpoint. A public repository shown on his portfolio (karamam.com). The newer version of this service runs as the Hugging Face Space karamAM/gpt2-demos.

## Files
- `README.md`: what the service is, how to run and deploy it, and how it differs from the Space.
- `app.py`: the web app and its one endpoint.
- `model.py`: loads the right model for each task and generates text.
- `GPT.py` and `GPTspam.py`: the GPT-2 network for text generation and for the spam classifier.
- `logging_utils.py`: logging and memory usage.
- `test_api.py`: a small script that calls a running service.
- `Dockerfile`, `start.sh`, `deploy.sh`: container image, local start and server deploy.
- `requirements.txt`: pinned packages.
- `.gitignore`: keeps weight files and the virtual environment out of git.
- `AGENTS.md`: this file.

## Folders
- `docs/`: guides for deploying on a DigitalOcean droplet, Docker commands and updating a running server.

## Rules
- Weight files (`model.pth`, `spam_model.pth`) are never committed.
