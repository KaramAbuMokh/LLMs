#!/usr/bin/env bash
set -euo pipefail

APP_NAME="gpt2-service"
OLD_NAME="gpt2-container"
NEW_NAME="gpt2-container-new"

# Tag with timestamp (or use `git rev-parse --short HEAD`)
TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="${APP_NAME}:${TAG}"

PORT_NEW=8081  # temporary port for health check

echo "==> Building ${IMAGE}"
docker build --pull -t "${IMAGE}" .

echo "==> Starting ${NEW_NAME} on :${PORT_NEW} for health check"
docker rm -f "${NEW_NAME}" >/dev/null 2>&1 || true
docker run -d \
  -p ${PORT_NEW}:8080 \
  -v /root/model.pth:/app/model.pth \
  -v /root/spam_model.pth:/app/spam_model.pth \
  -e CKPT_PATH=/app/model.pth \
  --name "${NEW_NAME}" \
  "${IMAGE}"

echo "==> Waiting for health..."
# Adjust /health path or timeout as needed
for i in {1..30}; do
  if curl -fsS "http://127.0.0.1:${PORT_NEW}/health" >/dev/null 2>&1; then
    echo "✓ Healthy"
    HEALTHY=1
    break
  fi
  sleep 1
done
if [ "${HEALTHY:-0}" -ne 1 ]; then
  echo "✗ New container failed health check. Keeping old container running."
  docker logs --tail=200 "${NEW_NAME}" || true
  docker rm -f "${NEW_NAME}" || true
  exit 1
fi

echo "==> Swapping traffic to new image"
# Stop & remove old, then run the new image on port 80
docker rm -f "${OLD_NAME}" >/dev/null 2>&1 || true

docker run -d \
  -p 80:8080 \
  -v /root/model.pth:/app/model.pth \
  -e CKPT_PATH=/app/model.pth \
  --name "${OLD_NAME}" \
  "${IMAGE}"

echo "==> Cleaning up temp container"
docker rm -f "${NEW_NAME}" || true

# Optional: prune old images but keep last 3 tagged ones of this app
echo "==> Pruning old images (keeping latest 3 ${APP_NAME} images)"
KEEP=3
IMAGES=($(docker images --format '{{.Repository}}:{{.Tag}} {{.CreatedAt}}' | grep "^${APP_NAME}:" | sort -rk2 | awk '{print $1}'))
for ((i=KEEP; i<${#IMAGES[@]}; i++)); do
  docker image rm -f "${IMAGES[$i]}" || true
done


echo "✅ Deploy complete: ${IMAGE} is live on port 80"
