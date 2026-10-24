# Deployment on DigitalOcean Droplet

Follow these steps to deploy the GPT-2 service using a Docker container on a DigitalOcean Droplet.

## 1. Create a Droplet
1. Log in to DigitalOcean and create a new droplet with the **Ubuntu 22.04 LTS** image.
2. Choose a **Basic** droplet plan with **at least 2 vCPUs and 4 GB RAM** for comfortable inference.
3. Select a region near your users, enable SSH authentication, and create the droplet.

## 2. Secure Networking
1. In the DigitalOcean dashboard, enable the firewall.
2. Allow inbound ports **22** (SSH) and **80** or **8080** (HTTP). Add **443** if HTTPS will be used.

## 3. SSH into the Droplet
```bash
ssh root@YOUR_DROPLET_IP
```

## 4. Update and Install Dependencies
```bash
apt update && apt upgrade -y
apt install -y docker.io git
systemctl enable --now docker
```

## 5. Clone the Repository and Provide Model Weights
```bash
git clone https://github.com/your-org/LLMs.git
cd LLMs
```
Upload your `.pth` model weights to the droplet (for example via `scp`) and note the path, such as `/root/model.pth`.

## 6. Configure Environment Variables
Configure the service using the following variables before running the container:
- `CKPT_PATH`: path to the model weights inside the container
- `MODEL_NAME`: model name to display in the API
- `TORCH_THREADS`: number of CPU threads for PyTorch
- `MAX_CTX`: maximum context length
- `API_KEY` *(optional)*: required API key for requests

## 7. Build the Docker Image
```bash
docker build -t gpt2-service .
```
This Dockerfile sets up the FastAPI app and GPT-2 model environment.

## 8. Run the Service
```bash
docker run -d \
  -p 80:8080 \
  -v /root/model.pth:/app/model.pth \
  -e CKPT_PATH=/app/model.pth \
  gpt2-service
```

## 9. Test the API
- Health check:
  ```bash
  curl http://YOUR_DROPLET_IP/health
  ```
- Text generation:
  ```bash
  curl -X POST "http://YOUR_DROPLET_IP/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The quick brown fox","max_new_tokens":12}'
  ```

## 10. Optional Production Hardening
- Use `ufw` to tighten firewall rules.
- Set up HTTPS with a reverse proxy such as Nginx or Caddy.
- Configure a process manager or orchestrator for restarts and scaling.

