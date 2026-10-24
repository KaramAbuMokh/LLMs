# Small Python image, same Python version as the original service
FROM python:3.11-slim

# Hugging Face runs the container as user id 1000, so create that user first
RUN useradd -m -u 1000 user
# Everything after this line runs as that user
USER user

# Settings: home folder, unbuffered logs, 2 CPU threads (cpu-basic has 2 vCPU), tokenizer cache folder
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_THREADS=2 \
    TIKTOKEN_CACHE_DIR=/home/user/tiktoken_cache

# Work inside the user's home folder
WORKDIR $HOME/app

# Install the Python packages first so this layer is cached when only code changes (CPU-only torch)
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download the GPT-2 tokenizer files now, so startup does not depend on an outside website
RUN python -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# The frozen contract and the shared encoders, then the two public encoder models at their pinned commits (MIT and Apache-2.0 licences)
COPY --chown=user retrieval_contracts ./retrieval_contracts
COPY --chown=user retrieval_encoders ./retrieval_encoders
RUN python -m retrieval_encoders.model_file_download --target /home/user/model_files

# The service code and the try-it page (weights and the retrieval index are NOT in the image: they come from private repos at startup with the HF_TOKEN secret)
COPY --chown=user gpt2_demo_service ./gpt2_demo_service

# Hugging Face expects the app on port 7860 (same as app_port in README.md)
EXPOSE 7860
# Start the web service with one worker so the models and the index are loaded once
CMD ["uvicorn", "gpt2_demo_service.web_api.app_factory:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
