# checked
import torch
import os
import torch.nn as nn
import logging
import numpy as np
from utils.model_exporter import export_model
from utils.evaluator import quick_f1

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

    # ✅ Count positives/negatives from generator to balance the loss
    num_pos, num_neg = 0, 0
    for _, y in train_loader():
        num_pos += (y == 1).sum().item()
        num_neg += (y == 0).sum().item()
    ratio = num_neg / max(1, num_pos)
    logging.info(f"[Info] pos_weight ratio = {ratio:.1f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    pos_weight = torch.tensor([ratio], device=device)
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

        # ✅ Updated F1 evaluation with threshold logging
        if return_best_f1:
            logging.info("[Eval] 🧪 Evaluating F1 for early stopping...")
            model.eval()
            metrics_dict = quick_f1(model, val_loader, device)
            precision  = metrics_dict["Precision"]
            recall     = metrics_dict["Recall"]
            f1_score_  = metrics_dict["F1"]     # avoid shadowing builtin f1_score
            th         = metrics_dict["th"]

            print(f"[Eval] th={th:.2f}  F1={f1_score_:.4f} | P={precision:.4f} | R={recall:.4f}")


            if f1_score_ > best_f1:
                best_f1 = f1_score_
                patience_counter = 0
                logging.info("🎯 New best F1 found!")
            else:
                patience_counter += 1
                if patience_counter >= config.get("early_stop_patience", 2):
                    logging.info("🛑 Early stopping triggered.")
                    break

    if not return_best_f1:
        logging.info("[LSTM] ✅ Skipping in-memory evaluation — handled separately.")
        try:
            export_model(model, "/app/models/lstm_rnn_trained_model.pth")
        except Exception as e:
            logging.error(f"[Export] ❌ Failed to export model: {e}")
        return model

    return best_f1 if return_best_f1 else model

