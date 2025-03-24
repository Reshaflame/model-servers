#!/bin/bash

cd /app

mkdir -p /app/models
echo "[Launcher] Starting end-to-end pipeline..."

# Step 1: Download datasets (if missing)
bash scripts/download_datasets.sh

# Step 2: Split dataset into chunks (only if chunks don't exist)
if [ ! -d "data/labeled_data/chunks" ]; then
    echo "[Step 0] Splitting large dataset into chunks..."
    python -c "from utils.datachunker import DataChunker; DataChunker('data/auth.txt.gz', output_dir='data/labeled_data/chunks').chunk_and_save()"
else
    echo "[Step 0] Skipping dataset split. Chunks already exist."
fi

# Step 3 onward:
echo "[Step 1] Preprocessing labeled data..."
python src/preprocess/labeledPreprocess.py

echo "[Step 2] Preprocessing unlabeled data..."
python src/preprocess/unlabeledPreprocess.py

echo "[Step 3] Cleaning up temporary chunks..."
python -c "from utils.cleanup import CleanupUtility; CleanupUtility.cleanup_all()"

echo "[Step 4] Training GRU..."
python src/pipeline/gru_pipeline.py

echo "[Step 5] Training LSTM+RNN..."
python src/pipeline/lstm_pipeline.py

echo "[Step 6] Training Transformer..."
python src/pipeline/tst_pipeline.py

echo "[Step 7] Training Isolation Forest..."
python src/pipeline/iso_pipeline.py

echo "[Step 8] Training Weighted Voting..."
python src/decision/weighted_voting.py

echo "[Step 9] Serving download page on port 8888..."
python src/utils/flask_server.py
