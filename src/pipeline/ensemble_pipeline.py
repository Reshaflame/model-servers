# checked
import numpy as np
from decision.weighted_voting import WeightedVoting

def run_ensemble_training():
    print("[Ensemble] Loading model predictions for ensemble training...")

    iso_preds = np.load("/app/models/preds/iso_preds.npy")
    gru_preds = np.load("/app/models/preds/gru_preds.npy")
    lstm_preds = np.load("/app/models/preds/lstm_preds.npy")
    transformer_preds = np.load("/app/models/preds/transformer_preds.npy")
    y_true = np.load("/app/models/preds/y_true.npy")

    predictions_list = [iso_preds, gru_preds, lstm_preds, transformer_preds]

    print(f"[Ensemble] Loaded {len(y_true)} samples from each model.")
    voting = WeightedVoting(num_models=4, weight_file="/app/models/ensemble_weights.json")
    voting.optimize_weights(predictions_list, y_true)
    print("[Ensemble] ✅ Ensemble weights saved to /app/models/ensemble_weights.json")


if __name__ == "__main__":
    run_ensemble_training()
