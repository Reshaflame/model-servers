# src/pipeline/iso_pipeline.py

from utils.tuning import SkoptTuner
from skopt.space import Real, Integer
import pandas as pd
from models.isolation_forest import IsolationForestModel
from preprocess.unlabeledPreprocess import preprocess_auth_data
from utils.metrics import Metrics
import os
import numpy as np


def run_iso_pipeline(preprocessed_path='data/sampled_data/auth.csv', 
                     preprocess=True, 
                     raw_data_path='data/auth.txt.gz',
                     sample_fraction=0.01):
    
    # Step 1: Preprocess if needed
    if preprocess:
        print("[Pipeline] Starting preprocessing...")
        preprocess_auth_data(file_path=raw_data_path, sample_fraction=sample_fraction)
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing CSV.")

    # Step 2: Load preprocessed data
    print("[Pipeline] Loading preprocessed data...")
    data = pd.read_csv(preprocessed_path)
    feature_columns = [col for col in data.columns if col.startswith(('auth_type_', 'logon_type_', 'auth_orientation_', 'success_'))]
    X = data[feature_columns].fillna(0)

    # Step 3: Dummy y_true (for now)
    y_dummy = [0] * len(X)  # Temporary until we have labeled data

    # Step 4: Train & Optimize
    print("[Pipeline] Optimizing hyperparameters with Skopt...")
    space = [
        Real(0.01, 0.2, name='contamination'),
        Integer(50, 300, name='n_estimators')
    ]

    tuner = SkoptTuner(IsolationForestModel, Metrics().compute_standard_metrics, space)
    best_params = tuner.optimize(X, y_dummy)

    # Update model with best params
    model = IsolationForestModel(contamination=best_params[0], n_estimators=int(best_params[1]))
    model.fit(X)
    
    # Step 5: Evaluate
    metrics = model.evaluate(X, y_dummy)
    print(f"[Pipeline] Metrics after optimization: {metrics}")

    # Step 6: Predict anomalies
    preds = model.predict(X)
    anomalies_detected = (preds == -1).sum()
    print(f"[Pipeline] Total anomalies detected: {anomalies_detected}")

    # Step 7: Export preds and y_true for ensemble training
    os.makedirs('/app/models/preds', exist_ok=True)
    np.save("/app/models/preds/iso_preds.npy", preds)
    np.save("/app/models/preds/y_true.npy", y_dummy)
    print("[Pipeline] Isolation Forest predictions exported for ensemble training.")

    # Optional: Save output
    data['anomaly'] = preds
    output_path = 'data/sampled_data/auth_with_anomalies.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"[Pipeline] Output saved to: {output_path}")

