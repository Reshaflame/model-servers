
# src/pipeline/iso_pipeline.py

from utils.tuning import SkoptTuner
from skopt.space import Real, Integer
from models.isolation_forest import IsolationForestModel
from preprocess.unlabeledPreprocess import preprocess_auth_data
from utils.metrics import Metrics
from utils.constants import CHUNKS_UNLABELED_PATH
import numpy as np
import os
import torch
from utils.chunked_dataset import ChunkedCSVDataset

def run_iso_pipeline(preprocess=True, raw_data_path='data/auth_quarter_01.txt.gz', sample_fraction=0.01):
    if preprocess:
        print("[Pipeline] Starting preprocessing...")
        preprocess_auth_data(file_path=raw_data_path, sample_fraction=sample_fraction)
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing chunked CSVs.")

    # Use disk-based chunk loader for large-scale inference
    chunk_dir = CHUNKS_UNLABELED_PATH
    chunk_dataset = ChunkedCSVDataset(
        chunk_dir=chunk_dir,
        label_column=None,
        batch_size=512,
        shuffle_files=False,
        binary_labels=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    X_all = []
    for features, _ in chunk_dataset.full_loader():
        X_all.append(features.cpu().numpy())
    X_all = np.concatenate(X_all, axis=0)

    # Dummy ground truth
    y_dummy = [0] * len(X_all)

    # Hyperparameter tuning
    print("[Pipeline] Optimizing hyperparameters with Skopt...")
    space = [
        Real(0.01, 0.2, name='contamination'),
        Integer(50, 300, name='n_estimators')
    ]

    tuner = SkoptTuner(IsolationForestModel, Metrics().compute_standard_metrics, space)
    best_params = tuner.optimize(X_all, y_dummy)

    # Train final model
    model = IsolationForestModel(contamination=best_params[0], n_estimators=int(best_params[1]))
    model.fit(X_all)

    # Inference
    preds = model.predict(X_all)
    anomalies_detected = (preds == -1).sum()
    print(f"[Pipeline] Total anomalies detected: {anomalies_detected}")

    os.makedirs('/app/models/preds', exist_ok=True)
    np.save("/app/models/preds/iso_preds.npy", preds)
    np.save("/app/models/preds/y_true.npy", y_dummy)
    print("[Pipeline] ✅ Isolation Forest predictions exported for ensemble training.")
