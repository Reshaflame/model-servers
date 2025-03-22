#!/bin/bash

mkdir -p /app/models
echo "[Launcher] Starting end-to-end pipeline..."

# Step 1: Download datasets (if missing)
bash /app/scripts/download_datasets.sh

# Step 2: Train GRU pipeline
echo "[Launcher] Training GRU model..."
python src/pipeline/gru_pipeline.py
echo "[Launcher] ✅ GRU model completed and exported."

# Step 3: Train LSTM+RNN pipeline
echo "[Launcher] Training LSTM+RNN model..."
python src/pipeline/lstm_pipeline.py
echo "[Launcher] ✅ LSTM+RNN model completed and exported."

# Step 4: Train Transformer pipeline
echo "[Launcher] Training Transformer model..."
python src/pipeline/tst_pipeline.py
echo "[Launcher] ✅ Transformer model completed and exported."

# Step 5: Launch Flask web server for model downloads
echo "[Launcher] Launching model download web server on port 8888..."
python src/utils/flask_server.py
