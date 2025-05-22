# src/decision/weighted_voting.py

import numpy as np
import json
from pathlib import Path
from skopt import gp_minimize
from utils.metrics import Metrics

class WeightedVoting:
    def __init__(self, num_models, weight_file="/app/models/ensemble_weights.json"):
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
        weighted_sum = sum([w * np.array(preds) for w, preds in zip(self.weights, predictions_list)])
        return (weighted_sum >= 0.5).astype(int)

    def optimize_weights(self, predictions_list, y_true):
        def objective(weights):
            norm_weights = [w / sum(weights) for w in weights]
            weighted_preds = sum([w * np.array(preds) for w, preds in zip(norm_weights, predictions_list)])
            y_pred = (weighted_preds >= 0.5).astype(int)
            metrics_result = self.metrics.compute_standard_metrics(y_true, y_pred)
            loss = -(metrics_result["F1"] + metrics_result["Accuracy"])
            return loss

        bounds = [(0.0, 1.0)] * self.num_models
        result = gp_minimize(objective, bounds, n_calls=20, random_state=42)
        self.weights = [w / sum(result.x) for w in result.x]
        self.save_weights()
        print(f"✅ Optimized Weights: {self.weights}")


# # =======================
# # ENTRYPOINT
# # =======================

# if __name__ == "__main__":
#     preds_dir = "/app/models/preds/"
#     labeled_csv = "/app/data/labeled_data/labeled_auth.csv"
    
#     # 1. Load ground truth
#     df = pd.read_csv(labeled_csv)
#     y_true = np.where(df['label'].values == -1, 0, 1)

#     # 2. Load model predictions (assuming all models dumped their outputs here)
#     model_preds = []
#     model_names = ["gru", "lstm", "transformer", "iso"]

#     for model in model_names:
#         path = os.path.join(preds_dir, f"{model}_preds.npy")
#         preds = np.load(path)
#         model_preds.append(preds)

#     # 3. Train the ensemble
#     print(f"Loaded predictions from: {model_names}")
#     ensemble = WeightedVoting(num_models=len(model_preds))
#     ensemble.optimize_weights(model_preds, y_true)
#     print(f"✅ Ensemble weights saved to {ensemble.weight_file}")
