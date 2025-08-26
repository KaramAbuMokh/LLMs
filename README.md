# GPT-2 Service

A lightweight FastAPI service for GPT-2 text generation with CPU optimization and quantization.

## Project Structure

```
gpt2_service/
├─ app.py          # FastAPI application with /generate endpoint
├─ model.py        # GPT-2 model loading, quantization, and generation
├─ requirements.txt # Python dependencies
├─ Dockerfile      # Container configuration
├─ start.sh        # Local development script
└─ README.md       # This file
```

## Features

- **CPU Optimized**: Uses PyTorch CPU builds and dynamic int8 quantization
- **Lightweight**: Minimal dependencies and small container size
- **Flexible**: Supports custom model weights (.pth files)
- **Secure**: Optional API key authentication
- **Production Ready**: Docker containerization and health checks

## Quick Start

### 1. Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your model path
export CKPT_PATH="path/to/your/model.pth"

# Run the service
python app.py
# OR use the start script
./start.sh
```

### 2. Docker Deployment

```bash
# Build the container
docker build -t gpt2-service .

# Run with your model mounted
docker run -p 8080:8080 -v /path/to/model.pth:/app/model.pth gpt2-service

# Or copy your model into the image (uncomment COPY line in Dockerfile)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CKPT_PATH` | `model.pth` | Path to your trained model weights |
| `MODEL_NAME` | `gpt2` | Base model configuration name |
| `TORCH_THREADS` | `1` | Number of CPU threads for PyTorch |
| `MAX_CTX` | `512` | Maximum context length (keeps it fast) |
| `API_KEY` | `""` | Optional API key for authentication |

## API Usage

### Health Check
```bash
curl http://localhost:8080/health
```

### Text Generation
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox",
    "task_name": "got",
    "max_new_tokens": 12,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50
  }'
```

### With API Key Authentication
```bash
export API_KEY="your_secret_key"

curl -H "x-api-key: your_secret_key" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello my name is","task_name":"spam","max_new_tokens":8}' \
     http://localhost:8080/generate
```

### Task Selection

The `/generate` endpoint supports two tasks controlled by the `task_name` field:

- `spam` &mdash; classify the provided text as spam or not spam
- `got` &mdash; generate Game of Thrones themed text

Omit `task_name` or use any other value to fall back to the default GPT-2 text generation.

## Model Requirements

Your `.pth` file should contain either:
- A complete model state dict
- A complete model object
- A state dict wrapped in a dictionary with a "state_dict" key

The service will automatically detect and load the appropriate format.

## Performance Tips

- **CPU Threads**: Adjust `TORCH_THREADS` based on your CPU cores
- **Context Length**: Lower `MAX_CTX` for faster inference
- **Quantization**: Already enabled for optimal CPU performance
- **Memory**: The service uses dynamic quantization to reduce memory usage

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure your `.pth` file path is correct
2. **Memory Issues**: Reduce `MAX_CTX` or `TORCH_THREADS`
3. **Slow Generation**: Check if quantization is working properly
4. **Port Already in Use**: Change the port in `start.sh` or Dockerfile

### Logs

The service will output detailed error messages for debugging. Check the console output for any error details.

## Production Deployment

For production use, consider:
- Using a reverse proxy (nginx)
- Setting up proper logging
- Monitoring with health checks
- Load balancing for multiple instances
- Proper security measures beyond simple API keys

