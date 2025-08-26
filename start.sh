#!/usr/bin/env bash
export CKPT_PATH=${CKPT_PATH:-model.pth}
export MODEL_NAME=${MODEL_NAME:-gpt2}
export API_KEY=${API_KEY:-""}   # set if you want simple auth
export TORCH_THREADS=1
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1 