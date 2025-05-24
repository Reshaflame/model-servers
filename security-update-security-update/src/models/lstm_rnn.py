# src/models/lstm_rnn.py

import torch
import os
import torch.nn as nn
import logging
import numpy as np
from utils.model_exporter import export_model
from utils.metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/lstm_training.log"),  # ← pod path to save log
        logging.StreamHandler()  # ← still prints to terminal
    ]
)

class LSTM_RNN_Hybrid(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(LSTM_RNN_Hybrid, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        rnn_out, _ = self.rnn(lstm_out)
        out = self.fc(rnn_out[:, -1, :])  # Use the last time step
        return self.sigmoid(out)
    
def save_checkpoint(model, optimizer, epoch, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ Ensure directory exists
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config
    }, path)
    logging.info(f"[Checkpoint] 💾 Saved checkpoint for epoch {epoch} at: {path}")

def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        return 0  # No checkpoint, start from epoch 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    logging.info(f"[Checkpoint] 🔁 Resumed from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]    

def train_model(config, train_loader, val_loader, input_size, return_best_f1=False):
    model = LSTM_RNN_Hybrid(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()
    
    checkpoint_path = f"/workspace/checkpoints/lstm_lr{config['lr']}_h{config['hidden_size']}_l{config['num_layers']}.pth"
    start_epoch = 0

    # Try to load from checkpoint
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, 3):
        model.train()
        logging.info(f"[LSTM] [Epoch {epoch+1}/3] 🔁 Training started...")
        batch_num = 0

        # 🔁 Reset the generator each epoch
        for features, labels in train_loader():
            batch_num += 1
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_num % 1000 == 0:
                logging.info(f"[LSTM]   └─ Batch {batch_num}: Loss = {loss.item():.6f}")
        
        logging.info(f"[LSTM] [Epoch {epoch+1}] ✅ Done.")
        save_checkpoint(model, optimizer, epoch + 1, config, checkpoint_path)


    if not return_best_f1:
        logging.info("[LSTM] ✅ Skipping in-memory evaluation — handled separately by evaluate_and_export.")
        try:
            export_model(model, "/app/models/lstm_rnn_trained_model.pth")
        except Exception as e:
            logging.error(f"[Export] ❌ Failed to export model: {e}")
        return model

    # === Evaluation after training ===
    logging.info("[Eval] 🔍 Running F1 evaluation for tuning (lightweight)...")
    model.eval()
    val_loss = 0
    tp = fp = fn = 0
    batch_id = 0

    with torch.no_grad():
        for features, labels in val_loader():
            batch_id += 1
            features, labels = features.to(device), labels.to(device)
            preds = model(features)
            loss = criterion(preds, labels)
            val_loss += loss.item()

            preds_np = (preds > 0.5).float().cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            tp += np.logical_and(preds_np == 1, labels_np == 1).sum()
            fp += np.logical_and(preds_np == 1, labels_np == 0).sum()
            fn += np.logical_and(preds_np == 0, labels_np == 1).sum()

            if batch_id % 100 == 0:
                logging.info(f"   └─ [Eval] Processed {batch_id} batches")

    val_loss /= max(1, batch_id)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logging.info(f"[Eval] ✅ Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {val_loss:.6f}")

    if return_best_f1:
        return f1
    return model



def evaluate_model(model, test_loader, device):
    model.eval()
    metrics = Metrics()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    results = metrics.compute_all(y_true, y_pred)
    logging.info("Metrics:", results)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    logging.info(f"Evaluation Accuracy: {accuracy:.4f}")
