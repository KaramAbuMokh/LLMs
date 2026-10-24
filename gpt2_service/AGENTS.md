# gpt2_service

Purpose: the first version of Karam Abu Mokh's GPT-2 service (Game of Thrones answers and spam classification behind one endpoint, FastAPI and Docker, deployed on a DigitalOcean droplet that is no longer running), kept for reference. It has no book search. The current service, with retrieval, is `../gpt2_service_huggingface_space/`.

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
