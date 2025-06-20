# src/pipeline/iso_hybrid_pipeline.py
"""
Isolation-Forest ➊ backbone ➋ bottleneck ➌ MLP-head fine-tune pipeline
---------------------------------------------------------------------
* Backbone (sklearn) is trained once on a subset of labelled chunks.
* A tiny PyTorch head is then fine-tuned on the full labelled stream.
"""

import json, logging, os, torch, numpy as np, pandas as pd
from glob import glob

from utils.constants import CHUNKS_LABELED_PATH
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from models.iso_backbone import IsoBackbone
from models.iso_hybrid   import IsoHybrid

# ───────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[IsoHybrid] %(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/iso_hybrid.log"),
        logging.StreamHandler()
    ]
)

# ──────────────────────────── main logic ────────────────────────────
def run_iso_hybrid_pipeline(preprocess: bool = False) -> None:
    if preprocess:
        logging.warning("🚧 Labeled data should already be pre-processed — skipping.")

    # ----------------------------------------------------------------
    # 0.  Load canonical feature order from JSON
    # ----------------------------------------------------------------
    with open("data/meta/expected_features.json") as f:
        expected_features = json.load(f)
    logging.info(f"🔍 Using {len(expected_features)} features from expected_features.json")

    # ----------------------------------------------------------------
    # 1.  Build chunked dataset (same code path as GRU, seq_len = 1)
    # ----------------------------------------------------------------
    chunk_dataset = SequenceChunkedDataset(
        batch_size      = 5_000,             # ≈ rows per “batch”
        shuffle_files   = False,
        label_column    = "label",
        sequence_length = 1,                 # <- makes it “tabular”
        device          = "cuda" if torch.cuda.is_available() else "cpu",
        expected_features = expected_features
    )

    # ----------------------------------------------------------------
    # 2.  Train Isolation-Forest backbone on a sample of chunks
    # ----------------------------------------------------------------
    max_backbone_chunks = 200
    X_train = []
    for i, (features, _) in enumerate(chunk_dataset.full_loader()):
        if i >= max_backbone_chunks:
            break
        X_train.append(features)
    X_train = np.concatenate(X_train, axis=0)
    logging.info(f"🧠 Backbone fit on {len(X_train):,} samples from {i+1} chunks")

    iso_backbone = IsoBackbone(contamination=0.05, n_estimators=100)
    iso_backbone.fit(X_train)
    os.makedirs("/app/models", exist_ok=True)
    iso_backbone.save("/app/models/iso_backbone.joblib")
    logging.info("✅ Saved iso_backbone.joblib")

    # ----------------------------------------------------------------
    # 3.  Fine-tune IsoHybrid (frozen backbone  + trainable head)
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IsoHybrid(IsoBackbone.load("/app/models/iso_backbone.joblib"),
                       freeze_bottleneck=False).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    epochs = 3
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb_np, yb_t in chunk_dataset.train_loader():
            # xb_np is numpy (N×F); labels are tensor already on correct device
            yb_t = yb_t.float().unsqueeze(1).to(device)  # [N] -> [N,1]
            optim.zero_grad()
            logits = model(xb_np)                        # forward handles numpy
            loss = criterion(logits, yb_t)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        logging.info(f"[Epoch {ep+1}/{epochs}] training loss = {running_loss:.6f}")

    torch.save(model.state_dict(), "/app/models/iso_hybrid.pth")
    logging.info("✅ Saved iso_hybrid.pth")

    # ----------------------------------------------------------------
    # 4.  Evaluate hybrid on full labelled stream
    # ----------------------------------------------------------------
    logging.info("🧪 Starting evaluation over full dataset…")
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for idx, (features_np, labels_t) in enumerate(chunk_dataset.full_loader(), 1):
            logits = model(features_np)                  # tensor [N,1]
            preds  = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().ravel()
            labels = labels_t.cpu().numpy().astype(int)

            tp += np.logical_and(preds == 1, labels == 1).sum()
            fp += np.logical_and(preds == 1, labels == 0).sum()
            fn += np.logical_and(preds == 0, labels == 1).sum()

            if idx % 100 == 0:
                logging.info(f"📥 Evaluated {idx} chunks…")

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    logging.info(f"📊 Hybrid metrics → P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")
    logging.info("🎯 IsoHybrid pipeline completed")


# --------------------------------------------------------------------
if __name__ == "__main__":
    run_iso_hybrid_pipeline()
