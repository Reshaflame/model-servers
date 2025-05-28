
# src/pipeline/iso_pipeline.py

from utils.tuning import SkoptTuner
from skopt.space import Real, Integer
from models.isolation_forest import IsolationForestModel
from utils.metrics import Metrics
from utils.constants import CHUNKS_LABELED_PATH
from utils.chunked_dataset import ChunkedCSVDataset
import numpy as np
import logging
import os
import torch

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
        logging.warning("🚧 Labeled data should already be preprocessed! Skipping manual preprocessing step.")
    
    logging.info("📦 Loading preprocessed labeled chunks...")
    chunk_dir = CHUNKS_LABELED_PATH
    chunk_dataset = ChunkedCSVDataset(
        chunk_dir=chunk_dir,
        chunk_size=5000,
        label_col='label',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        expected_features=None
    )

    # === Step 1: Load Features and Labels from Disk Chunks ===
    X_all, y_true = [], []
    for features, labels in chunk_dataset.full_loader():
        X_all.append(features.cpu().numpy())
        y_true.append(labels.cpu().numpy())
    X_all = np.concatenate(X_all, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    logging.info(f"✅ Loaded {len(X_all)} samples from disk.")

    # === Step 2: Hyperparameter Optimization ===
    logging.info("🎯 Starting hyperparameter tuning with Skopt...")
    space = [
        Real(0.01, 0.2, name='contamination'),
        Integer(50, 300, name='n_estimators')
    ]

    tuner = SkoptTuner(IsolationForestModel, Metrics().compute_standard_metrics, space)
    best_params = tuner.optimize(X_all, y_true)
    contamination, n_estimators = best_params
    logging.info(f"🏆 Best params found: contamination={contamination}, n_estimators={n_estimators}")

    # === Step 3: Train Final Model ===
    model = IsolationForestModel(contamination=contamination, n_estimators=int(n_estimators))
    model.fit(X_all)
    logging.info("🧠 Isolation Forest model trained.")

    # === Step 4: Evaluate and Save ===
    metrics = model.evaluate(X_all, y_true)
    logging.info(f"📊 Evaluation Metrics: {metrics}")

    y_pred = np.where(model.predict(X_all) == -1, 1, 0)
    anomalies_detected = y_pred.sum()
    logging.info(f"🚨 Total anomalies detected: {anomalies_detected} / {len(y_pred)}")

    os.makedirs('/app/models/preds', exist_ok=True)
    np.save("/app/models/preds/iso_preds.npy", y_pred)
    np.save("/app/models/preds/y_true.npy", y_true)
    logging.info("💾 Predictions and ground truth saved to /app/models/preds")

    logging.info("✅ Isolation Forest pipeline completed successfully.")
