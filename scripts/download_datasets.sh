#!/bin/bash

echo "[Downloader] Creating /app/data directory..."
mkdir -p /app/data

echo "[Downloader] Downloading auth.txt.gz..."
gdown --id 1ltgSaY4Am1mX6wwpKyOgGxL9XgTBUrm1 -O /app/data/auth.txt.gz

echo "[Downloader] Downloading redteam.txt.gz..."
gdown --id 19wVQKYQhgj3ziLvXz1aNY10x4Qt3UxBC -O /app/data/redteam.txt.gz

echo "[Downloader] ✅ All datasets downloaded into /app/data"
