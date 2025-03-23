#!/bin/bash

mkdir -p /app/models
echo "[Launcher] Starting end-to-end pipeline..."

# Step 1: Download datasets (if missing)
bash /app/scripts/download_datasets.sh

# Step 2 onward: relative paths from /app
echo "[Step 1] Preprocessing labeled data..."
python src/preprocess/labeledPreprocess.py

echo "[Step 2] Preprocessing unlabeled data..."
python src/preprocess/unlabeledPreprocess.py

echo "[Step 3] Training GRU..."
python src/pipeline/gru_pipeline.py

echo "[Step 4] Training LSTM+RNN..."
python src/pipeline/lstm_pipeline.py

echo "[Step 5] Training Transformer..."
python src/pipeline/tst_pipeline.py

echo "[Step 6] Training Isolation Forest..."
python src/pipeline/iso_pipeline.py

echo "[Step 7] Training Weighted Voting..."
python src/decision/weighted_voting.py

echo "[Step 8] Serving download page on port 8888..."
python src/utils/flask_server.py
