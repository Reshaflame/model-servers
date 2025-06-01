import pandas as pd
import torch
import numpy as np
from models.gru import GRUAnomalyDetector
from models.lstm_rnn import LSTM_RNN_Hybrid
from models.isolation_forest import IsolationForestModel
from utils.constants import CHUNKS_LABELED_PATH
import json
import os
import joblib

# --- Load chunk data ---
chunk_path = os.path.join(CHUNKS_LABELED_PATH, "chunk_0_labeled.csv")
df = pd.read_csv(chunk_path)
df = df.dropna()
labels = df['label'].values
features = df.drop(columns=["label"]).select_dtypes(include=[np.number])

with open("data/expected_features.json") as f:
    expected_features = json.load(f)

# --- Align with expected features ---
for col in expected_features:
    if col not in features.columns:
        features[col] = 0
features = features[expected_features].astype(np.float32)

# --- Torch tensor and reshape ---
X = torch.tensor(features.values, dtype=torch.float32)
X_seq = X.unsqueeze(1).repeat(1, 10, 1)  # shape [B, 10, F]

# --- Load GRU ---
input_size = X.shape[1]
gru_model = GRUAnomalyDetector(input_size=input_size, hidden_size=64, num_layers=1)
gru_model.load_state_dict(torch.load("models/gru_trained_model.pth", map_location="cpu"))
gru_model.eval()

# --- Load LSTM ---
lstm_model = LSTM_RNN_Hybrid(input_size=input_size, hidden_size=64, num_layers=1)
lstm_model.load_state_dict(torch.load("models/lstm_rnn_trained_model.pth", map_location="cpu"))
lstm_model.eval()

# --- Load ISO Forest ---
iso_model = joblib.load("models/isolation_forest_model.joblib")

# --- Predict ---
with torch.no_grad():
    gru_scores = torch.sigmoid(gru_model(X_seq)).squeeze().numpy()
    lstm_scores = torch.sigmoid(lstm_model(X_seq)).squeeze().numpy()

iso_scores_raw = iso_model.decision_function(X.numpy())
iso_scores = (iso_scores_raw - np.min(iso_scores_raw)) / (np.ptp(iso_scores_raw) + 1e-9)  # normalize 0-1

# --- Ensemble (same weights as server) ---
W_GRU, W_LSTM, W_ISO = 0.4167, 0.4167, 0.1666
ensemble_scores = W_GRU * gru_scores + W_LSTM * lstm_scores + W_ISO * iso_scores
preds = ensemble_scores > 0.5

# --- Show some rows ---
print("🔎 Top 10 sample predictions:")
for i in range(10):
    print(f"Row {i} | Label: {labels[i]} | GRU: {gru_scores[i]:.4f} | LSTM: {lstm_scores[i]:.4f} | ISO: {iso_scores[i]:.4f} | Ensemble: {ensemble_scores[i]:.4f} | Predicted: {preds[i]}")
