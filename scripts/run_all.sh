# Fix pipeline imports to remove old preprocess references
# Update `run_all.sh` script call for DataChunker and CleanupUtility

updates = [
    {
        "pattern": ".*",
        "replacement": """
#!/bin/bash

cd /app

mkdir -p /app/models
echo \"[Launcher] Starting end-to-end pipeline...\"

# Step 1: Download datasets (if missing)
bash scripts/download_datasets.sh

# Step 2: Split dataset into chunks (only if chunks don't exist)
if [ ! -d \"data/labeled_data/chunks\" ]; then
    echo \"[Step 0] Splitting large dataset into chunks...\"
    python -c \"from utils.datachunker import DataChunker; DataChunker('data/auth.txt.gz', output_dir='data/labeled_data/chunks').chunk_and_save()\"
else
    echo \"[Step 0] Skipping dataset split. Chunks already exist.\"
fi

# Step 3 onward:
echo \"[Step 1] Preprocessing labeled data...\"
python src/preprocess/labeledPreprocess.py

echo \"[Step 2] Preprocessing unlabeled data...\"
python src/preprocess/unlabeledPreprocess.py

# Step 4: Cleanup chunk files
echo \"[Step 3] Cleaning up temporary chunks...\"
python -c \"from utils.cleanup import CleanupUtility; CleanupUtility.cleanup_all()\"

# Step 5: Training GRU
echo \"[Step 4] Training GRU...\"
python src/pipeline/gru_pipeline.py

# Step 6: Training LSTM+RNN
echo \"[Step 5] Training LSTM+RNN...\"
python src/pipeline/lstm_pipeline.py

# Step 7: Training Transformer
echo \"[Step 6] Training Transformer...\"
python src/pipeline/tst_pipeline.py

# Step 8: Training Isolation Forest
echo \"[Step 7] Training Isolation Forest...\"
python src/pipeline/iso_pipeline.py

# Step 9: Training Weighted Voting
echo \"[Step 8] Training Weighted Voting...\"
python src/decision/weighted_voting.py

# Step 10: Serve download page
echo \"[Step 9] Serving download page on port 8888...\"
python src/utils/flask_server.py
"""
    }
]
