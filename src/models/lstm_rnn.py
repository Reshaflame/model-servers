import torch
import os
import torch.nn as nn
import logging
import numpy as np
from utils.model_exporter import export_model
from utils.metrics import Metrics

class LSTM_RNN_Hybrid(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(LSTM_RNN_Hybrid, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        rnn_out, _ = self.rnn(lstm_out)
        out = self.fc(rnn_out[:, -1, :])
        return out  # logits (no sigmoid)

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

def train_model(config, train_loader, val_loader, input_size, return_best_f1=False):
    
    # ✅ Init logging only during training
    os.makedirs("/workspace/logs", exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [LSTM] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler("/workspace/logs/lstm_training.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    model = LSTM_RNN_Hybrid(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    pos_weight = torch.tensor([100.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_path = f"/workspace/checkpoints/lstm_lr{config['lr']}_h{config['hidden_size']}_l{config['num_layers']}.pth"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    epochs = config.get("epochs", 3)
    best_f1 = 0
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        logging.info(f"[LSTM] [Epoch {epoch+1}/{epochs}] 🔁 Training started...")
        for batch_num, (features, labels) in enumerate(train_loader(), 1):
            features, labels = features.to(device), labels.to(device)
            if labels.dim() == 3:
                labels = labels.squeeze(-1)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if batch_num % 1000 == 0:
                logging.info(f"[LSTM]   └─ Batch {batch_num}: Loss = {loss.item():.6f}")

        logging.info(f"[LSTM] [Epoch {epoch+1}] ✅ Done.")
        save_checkpoint(model, optimizer, epoch + 1, config, checkpoint_path)

    if not return_best_f1:
        logging.info("[LSTM] ✅ Skipping in-memory evaluation — handled separately.")
        try:
            export_model(model, "/app/models/lstm_rnn_trained_model.pth")
        except Exception as e:
            logging.error(f"[Export] ❌ Failed to export model: {e}")
        return model

    # === Lightweight F1 Eval ===
    logging.info("[Eval] 🔍 Running F1 evaluation...")
    model.eval()
    val_loss = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for batch_id, (features, labels) in enumerate(val_loader(), 1):
            features, labels = features.to(device), labels.to(device)
            if labels.dim() == 3:
                labels = labels.squeeze(-1)

            logits = model(features)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            tp += np.logical_and(preds == 1, labels_np == 1).sum()
            fp += np.logical_and(preds == 1, labels_np == 0).sum()
            fn += np.logical_and(preds == 0, labels_np == 1).sum()

            if batch_id % 100 == 0:
                logging.info(f"   └─ [Eval] Processed {batch_id} batches")

    val_loss /= max(1, batch_id)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    logging.info(f"[Eval] ✅ Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {val_loss:.6f}")
    return f1 if return_best_f1 else model
