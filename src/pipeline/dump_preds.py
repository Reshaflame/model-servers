# src/pipeline/dump_preds.py
import torch, numpy as np, json, argparse, pathlib
from utils.chunked_dataset import SequenceChunkedDataset     # streams from CSV
from utils.evaluator       import batched_logits
from models.gru_hybrid     import GRUAnomalyDetector
from models.lstm_rnn       import LSTMRNNBackbone
from models.isolation_forest import load_iforest

FEATS = json.load(open("data/meta/expected_features.json"))
CHUNKS = "data/chunks_labeled_phase2"
OUT = pathlib.Path("models/preds");  OUT.mkdir(parents=True, exist_ok=True)

def save(name, arr): np.save(OUT / f"{name}.npy", arr.astype(np.float32))

def main():
    # ---------- LSTM loader (seq_len=10) ----------
    l_loader = SequenceChunkedDataset(CHUNKS, seq_len=10,
                                      feature_cols=FEATS).full_loader()
    lstm = LSTMRNNBackbone.load_from_checkpoint("models/lstm_rnn_trained_model.pth")
    y_l, p_l = batched_logits(lstm, l_loader)
    save("lstm_preds", p_l)          # len = N-9
    save("y_true",      y_l)

    # ---------- GRU & IF loaders (seq_len=1) ----------
    s_loader = SequenceChunkedDataset(CHUNKS, seq_len=1,
                                      feature_cols=FEATS).full_loader()

    gru  = GRUAnomalyDetector.load_from_checkpoint("models/gru_trained_model.pth")
    _, p_g = batched_logits(gru, s_loader)
    save("gru_preds", p_g[9:])       # align: drop first 9

    iso  = load_iforest("models/iso_backbone.joblib")
    _, p_i = batched_logits(iso, s_loader)
    save("iso_preds", p_i[9:])       # align
