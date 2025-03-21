# src/decision/weighted_voting.py

import numpy as np
import json
from pathlib import Path
from skopt import gp_minimize
from src.utils.metrics import Metrics

class WeightedVoting:
    def __init__(self, num_models, weight_file="ensemble_weights.json"):
        self.num_models = num_models
        self.weight_file = weight_file
        self.metrics = Metrics()
        self.weights = self._load_weights()

    def _load_weights(self):
        if Path(self.weight_file).exists():
            with open(self.weight_file, "r") as f:
                return json.load(f)
        return [1.0 / self.num_models] * self.num_models  # default uniform weights

    def save_weights(self):
        with open(self.weight_file, "w") as f:
            json.dump(self.weights, f)

    def predict(self, predictions_list):
        """
        predictions_list: List of binary predictions [model1_preds, model2_preds, ...]
        """
        weighted_sum = sum([w * np.array(preds) for w, preds in zip(self.weights, predictions_list)])
        return (weighted_sum >= 0.5).astype(int)

    def optimize_weights(self, predictions_list, y_true, anomaly_ranges=None, pred_ranges=None):
        """
        Optimize weights based on TaPR + F1 as objective.
        """
        def objective(weights):
            norm_weights = [w / sum(weights) for w in weights]  # Normalize
            weighted_preds = sum([w * np.array(preds) for w, preds in zip(norm_weights, predictions_list)])
            y_pred = (weighted_preds >= 0.5).astype(int)

            metrics_result = self.metrics.compute_all(
                y_true, y_pred, anomaly_ranges, pred_ranges
            )

            # Example: prioritize TaPR + F1 (customize as you want)
            loss = -(metrics_result["TaPR"] + metrics_result["F1"])
            return loss

        # Search space: weights between [0,1] for each model
        bounds = [(0.0, 1.0)] * self.num_models
        result = gp_minimize(objective, bounds, n_calls=30, random_state=42)
        self.weights = [w / sum(result.x) for w in result.x]  # Normalize and save
        self.save_weights()
        print(f"Optimized Weights: {self.weights}")
