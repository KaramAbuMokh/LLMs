FROM python:3.11-slim

# (Optional) if you know your CPU wheels, pre-set index here for faster installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    TORCH_THREADS=1 \
    MAX_CTX=512

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# copy code and your weights (rename your .pth to model.pth or mount at runtime)
# include all Python modules required by the service
COPY *.py ./
COPY model.pth ./model.pth  # uncomment if bundling weights into image
COPY spam_model.pth ./spam_model.pth

# health & port
EXPOSE 8080

# small start script to warm-up on first request
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 
