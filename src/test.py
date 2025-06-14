import pandas as pd
import torch
import numpy as np
import json
import os
import joblib
from collections import Counter

from models.gru_hybrid import GRUAnomalyDetector
from models.lstm_rnn import LSTM_RNN_Hybrid
from models.isolation_forest import IsolationForestModel
from utils.constants import CHUNKS_LABELED_PATH

# === Load chunk ===
chunk_path = os.path.join(CHUNKS_LABELED_PATH, "chunk_0_labeled.csv")
df = pd.read_csv(chunk_path)

print("\n📊 Label Distribution:")
print(df["label"].value_counts())

# === Feature preprocessing ===
labels = df['label'].values
features = df.drop(columns=["label"]).select_dtypes(include=[np.number])

with open("data/expected_features.json") as f:
    expected_features = json.load(f)

# Pad missing features
for col in expected_features:
    if col not in features.columns:
        features[col] = 0
features = features[expected_features].astype(np.float32)

# === Inspect feature stats ===
print("\n📈 Feature Stats:")
print(features.describe().T[["mean", "std", "min", "max"]])

# === Torch tensor and sequence reshaping ===
X = torch.tensor(features.values, dtype=torch.float32)
X_seq = X.unsqueeze(1).repeat(1, 10, 1)

# === Load GRU ===
input_size = X.shape[1]
gru_model = GRUAnomalyDetector(input_size=input_size, hidden_size=64, num_layers=1)
gru_model.load_state_dict(torch.load("models/gru_trained_model.pth", map_location="cpu"))
gru_model.eval()

# === Load LSTM ===
lstm_model = LSTM_RNN_Hybrid(input_size=input_size, hidden_size=64, num_layers=1)
lstm_model.load_state_dict(torch.load("models/lstm_rnn_trained_model.pth", map_location="cpu"))
lstm_model.eval()

# === Load ISO Forest ===
iso_model = joblib.load("models/isolation_forest_model.joblib")

# === Run Predictions ===
with torch.no_grad():
    gru_logits = gru_model(X_seq)
    lstm_logits = lstm_model(X_seq)
    gru_scores = 1 - torch.sigmoid(gru_model(X_seq)).squeeze().numpy()
    lstm_scores = 1 - torch.sigmoid(lstm_model(X_seq)).squeeze().numpy()

iso_raw = iso_model.decision_function(X.numpy())
iso_scores = (iso_raw - np.min(iso_raw)) / (np.ptp(iso_raw) + 1e-9)

# === Print score distributions ===
print("\n📊 GRU Score Distribution:")
print(pd.Series(gru_scores).describe())
print("\n📊 LSTM Score Distribution:")
print(pd.Series(lstm_scores).describe())
print("\n📊 ISO Score Distribution:")
print(pd.Series(iso_scores).describe())

# === Check predictions on 0 vs 1 labels ===
def evaluate_threshold(scores, name):
    print(f"\n🔍 {name} Prediction Breakdown:")
    pred_labels = scores > 0.5
    combined = list(zip(labels, pred_labels))
    count = Counter((int(l), int(p)) for l, p in combined)
    for (true, pred), c in sorted(count.items()):
        print(f"True={true} Pred={pred} Count={c}")

evaluate_threshold(gru_scores, "GRU")
evaluate_threshold(lstm_scores, "LSTM")
evaluate_threshold(iso_scores, "ISO (threshold=0.5 normalized)")

# === Ensemble ===
W_GRU, W_LSTM, W_ISO = 0.4167, 0.4167, 0.1666
ensemble_scores = W_GRU * gru_scores + W_LSTM * lstm_scores + W_ISO * iso_scores
ensemble_preds = ensemble_scores > 0.5
print("\n📊 Ensemble Score Distribution:")
print(pd.Series(ensemble_scores).describe())
evaluate_threshold(ensemble_scores, "Ensemble")
