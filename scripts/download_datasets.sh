#!/bin/bash

mkdir -p data

# Check and download auth.txt.gz if missing
if [ ! -f data/auth.txt.gz ]; then
    echo "[Downloader] Downloading auth.txt.gz..."
    gdown --id 1ltgSaY4Am1mX6wwpKyOgGxL9XgTBUrm1 -O data/auth.txt.gz
else
    echo "[Downloader] auth.txt.gz already exists, skipping download."
fi

# Check and download redteam.txt.gz if missing
if [ ! -f data/redteam.txt.gz ]; then
    echo "[Downloader] Downloading redteam.txt.gz..."
    gdown --id 19wVQKYQhgj3ziLvXz1aNY10x4Qt3UxBC -O data/redteam.txt.gz
else
    echo "[Downloader] redteam.txt.gz already exists, skipping download."
fi

if [ $? -ne 0 ]; then
    echo "[Downloader] ❌ Download failed!"
    exit 1
fi

echo "[Downloader] ✅ Dataset download step completed."
