#!/bin/bash

# Step 1: Start dockerd if not already running
if ! pgrep -x "dockerd" > /dev/null
then
    echo "[Launcher] Starting dockerd..."
    sudo dockerd > /dev/null 2>&1 &
    DOCKERD_STARTED=1
else
    echo "[Launcher] dockerd already running."
fi

# Step 2: Wait until Docker daemon is fully ready
echo "[Launcher] Waiting for Docker daemon to be ready..."
until docker info > /dev/null 2>&1
do
    sleep 1
done

# Step 3: Launch Docker Compose orchestration
echo "[Launcher] Running docker-compose up..."
docker compose up --build

# Optional: Stop dockerd automatically after shutdown
if [ "$DOCKERD_STARTED" = "1" ]; then
    echo "[Launcher] Stopping dockerd..."
    sudo pkill dockerd
fi
