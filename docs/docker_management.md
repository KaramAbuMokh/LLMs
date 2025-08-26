# Docker Management Commands

This document lists common Docker commands for building, running, and managing the GPT-2 service container.

## Build the Image
```bash
docker build -t gpt2-service .
```
Builds the image defined by the repository's `Dockerfile` and tags it as `gpt2-service`.

## Run the Service
```bash
docker run -d \
  -p 80:8080 \
  -v /root/model.pth:/app/model.pth \
  -e CKPT_PATH=/app/model.pth \
  --name gpt2-container \
  gpt2-service
```
Runs the container in detached mode with the required port mapping, model weights, and environment variable. The container is named `gpt2-container` for easier management.

## Check Running Containers
```bash
docker ps
```
Shows running containers. Add `-a` to include stopped containers.

## View Logs
```bash
docker logs gpt2-container
```
Displays the container's stdout/stderr logs. Add `-f` to follow in real time.

## Stop the Service
```bash
docker stop gpt2-container
```
Gracefully stops the running container.

## Start the Service Again
```bash
docker start gpt2-container
```
Starts a stopped container without rebuilding the image.

## Remove the Container
```bash
docker rm gpt2-container
```
Deletes the container. Use `-f` to force removal if it is running.

## Remove the Image
```bash
docker rmi gpt2-service
```
Deletes the image. Ensure no containers are using it before removal.

## Check Disk Usage
```bash
docker system df
```
Displays disk usage by images, containers, and volumes.

