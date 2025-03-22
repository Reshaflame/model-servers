#!/bin/bash

echo "[Launcher] Starting pipeline setup..."

# Step 1: Download datasets (if missing)
bash /app/scripts/download_datasets.sh

# Step 2: Prompt user to select model
echo "[Launcher] Choose a model pipeline to run:"
echo "1. GRU Pipeline"
echo "2. LSTM+RNN Pipeline"
echo "3. Transformer Pipeline"
read -p "Enter choice [1-3]: " model_choice

case $model_choice in
    1)
        echo "[Launcher] Running GRU Pipeline..."
        python src/pipeline/gru_pipeline.py
        ;;
    2)
        echo "[Launcher] Running LSTM+RNN Pipeline..."
        python src/pipeline/lstm_pipeline.py
        ;;
    3)
        echo "[Launcher] Running Transformer Pipeline..."
        python src/pipeline/tst_pipeline.py
        ;;
    *)
        echo "[Launcher] Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "[Launcher] ✅ Process completed!"
