# src/models/transformer.py

import os
import torch
import torch.nn as nn
import numpy as np
import logging
from utils.model_exporter import export_model
from utils.metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/transformer_training.log"),
        logging.StreamHandler()
    ]
)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        seq_len = x.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :]
        x = self.input_projection(x) + positional_encoding
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

def save_checkpoint(model, optimizer, epoch, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config
    }, path)
    logging.info(f"[Checkpoint] 💾 Saved checkpoint for epoch {epoch} at: {path}")

def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    logging.info(f"[Checkpoint] 🔁 Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]

def train_transformer(config, train_loader, val_loader, input_size, return_best_f1=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        input_size=input_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = f"/workspace/checkpoints/tst_lr{config['lr']}_d{config['d_model']}_e{config['num_encoder_layers']}.pth"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, 3):
        model.train()
        logging.info(f"[TST] [Epoch {epoch+1}/3] 🔁 Training started...")
        batch_id = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            outputs = outputs.view(-1)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_id += 1
            if batch_id % 500 == 0:
                logging.info(f"[TST]   └─ Batch {batch_id}: Loss = {loss.item():.6f}")
        save_checkpoint(model, optimizer, epoch + 1, config, checkpoint_path)
        logging.info(f"[TST] [Epoch {epoch+1}] ✅ Done.")


    if not return_best_f1:
        logging.info("[TST] ✅ Skipping in-memory evaluation — handled separately by evaluate_and_export.")
        export_model(model, "/app/models/transformer_trained_model.pth")
        return model

    # Lightweight batchwise evaluation to avoid OOM
    logging.info("[Eval] 🔍 Running F1 evaluation for tuning (lightweight)...")
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    val_loss, tp, fp, fn, batch_id = 0, 0, 0, 0, 0

    with torch.no_grad():
        for features, labels in val_loader:
            batch_id += 1
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            outputs = outputs.view(-1)           # 👈 reshape logits
            labels = labels.view(-1)             # 👈 reshape labels
            loss = criterion(outputs, labels)    # logits go into BCEWithLogitsLoss
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)       # 👈 apply sigmoid here to get probs
            preds_np = (preds > 0.5).float().cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            tp += np.logical_and(preds_np == 1, labels_np == 1).sum()
            fp += np.logical_and(preds_np == 1, labels_np == 0).sum()
            fn += np.logical_and(preds_np == 0, labels_np == 1).sum()

            if batch_id % 100 == 0:
                logging.info(f"   └─ [Eval] Processed {batch_id} batches")


    val_loss /= max(1, batch_id)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    logging.info(f"[Eval] ✅ Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {val_loss:.6f}")

    if return_best_f1:
        logging.info(f"[Eval] Returning F1 score: {f1:.4f}")
        return f1
    return model

