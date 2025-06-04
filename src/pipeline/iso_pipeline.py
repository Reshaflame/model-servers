# src/pipeline/iso_pipeline.py
# checked

from models.isolation_forest import IsolationForestModel
from utils.constants import CHUNKS_LABELED_PATH
from utils.chunked_dataset import ChunkedCSVDataset
import numpy as np
import logging
import os
import torch
import joblib
import pandas as pd
from glob import glob

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='[IsolationForest] %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/workspace/logs/isolation_forest.log"),
        logging.StreamHandler()
    ]
)

def run_iso_pipeline(preprocess=False):
    if preprocess:
        logging.warning("🚧 Labeled data should already be preprocessed! Skipping manual preprocessing.")
    
    logging.info("📦 Loading preprocessed labeled chunks...")
    chunk_dir = CHUNKS_LABELED_PATH

    # === Step 0: Infer numeric features like GRU does ===
    first_chunk_path = glob(os.path.join(chunk_dir, "*.csv"))[0]
    df_sample = pd.read_csv(first_chunk_path)
    expected_features = df_sample.drop(columns=['label']).select_dtypes(include=['number']).columns.tolist()
    logging.info(f"🔍 Using {len(expected_features)} numeric features for Isolation Forest training.")

    chunk_dataset = ChunkedCSVDataset(
        chunk_dir=chunk_dir,
        chunk_size=5000,
        label_col='label',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        expected_features=expected_features
    )

    # === Step 1: Train on a sample (to avoid OOM) ===
    X_train = []
    max_train_chunks = 200
    logging.info(f"🧠 Sampling first {max_train_chunks} chunks for training...")

    for i, (features, _) in enumerate(chunk_dataset.full_loader()):
        if i >= max_train_chunks:
            break
        X_train.append(features)
    
    X_train = np.concatenate(X_train, axis=0)
    logging.info(f"✅ Training set loaded with {len(X_train)} samples.")

    model = IsolationForestModel(contamination=0.05, n_estimators=100)
    model.fit(X_train)

    os.makedirs("/workspace/checkpoints", exist_ok=True)
    joblib.dump(model.model, "/workspace/checkpoints/isolation_forest_model.joblib")
    logging.info("✅ Isolation Forest model trained and saved.")

    # === Step 2: Evaluate in streaming mode without growing memory ===
    logging.info("🧪 Starting evaluation on full dataset...")

    tp = fp = fn = 0
    for i, (features, labels) in enumerate(chunk_dataset.full_loader(), 1):
        preds = model.predict_labels(features)
        labels_np = np.array(labels)
        preds_np = np.array(preds)

        tp += np.logical_and(preds_np == 1, labels_np == 1).sum()
        fp += np.logical_and(preds_np == 1, labels_np == 0).sum()
        fn += np.logical_and(preds_np == 0, labels_np == 1).sum()

        if i % 100 == 0:
            logging.info(f"📥 Evaluated {i} chunks...")

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "precision": precision,
        "recall": recall,
        "F1": f1
    }

    logging.info(f"📊 Final Evaluation Metrics: {metrics}")

    logging.info("🎯 Isolation Forest pipeline completed successfully.")
