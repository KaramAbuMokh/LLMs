# Deployment on DigitalOcean Droplet

Follow these steps to deploy the GPT-2 service new updates on main branch

## 1. Stop and remove the old container

```bash
docker stop gpt2-container
docker rm gpt2-container
```

## 2. Pull the latest code
```bash
git pull origin main
```

## 3. Rebuild the image
```bash
docker build -t gpt2-service .
```

## 4. Run the new container
```bash
docker run -d \
  -p 80:8080 \
  -v /root/model.pth:/app/model.pth \
  -e CKPT_PATH=/app/model.pth \
  --name gpt2-container \
  gpt2-service
```
