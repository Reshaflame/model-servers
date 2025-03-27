
#!/bin/bash

cd /app

mkdir -p /app/models
echo "[Launcher] Starting end-to-end pipeline..."

# Step 0: Download datasets (if missing)
bash scripts/download_datasets.sh

# Step 0.1: Split dataset into chunks (only if chunks don't exist)
if [ ! -d "data/shared_chunks" ]; then
    echo "[Step 0] Splitting large dataset into chunks..."
    python -c "from utils.datachunker import DataChunker; DataChunker('data/auth_quarter_01.txt.gz', output_dir='data/shared_chunks').chunk_and_save()"
else
    echo "[Step 0] Skipping dataset split. Chunks already exist."
fi

echo "[Step 0.1] Deleting auth_quarter_01.txt.gz after chunking..."
python -c "from utils.cleanup import CleanupUtility; CleanupUtility.cleanup_raw_auth()"

# Step 3 onward:
echo "[Step 1] Preprocessing labeled data..."
python src/preprocess/labeledPreprocess.py

echo "[Step 2] Preprocessing unlabeled data..."
python src/preprocess/unlabeledPreprocess.py

echo "[Step 2.1] Deleting raw chunks after preprocessing..."
python -c "from utils.cleanup import CleanupUtility; CleanupUtility.cleanup_raw_chunks()"

# Move preprocessed chunked CSVs to /app/models for manual download (optional reuse)
echo "[Export] Moving labeled/unlabeled preprocessed chunks to /app/models..."

mkdir -p /app/models/chunks_labeled
mkdir -p /app/models/chunks_unlabeled

cp data/labeled_data/chunks/*.csv /app/models/chunks_labeled/
cp data/preprocessed_unlabeled/chunks/*.csv /app/models/chunks_unlabeled/

echo "[Export] ✅ Preprocessed chunks copied to /app/models/chunks_labeled and chunks_unlabeled"

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
