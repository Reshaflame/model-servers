# checked
import numpy as np
import torch
import torch.nn as nn
from utils.metrics import Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.evaluator import quick_f1


metrics = Metrics()

class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRUAnomalyDetector, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.contiguous()
        h, _ = self.gru(x)
        out = self.fc(self.dropout(h[:, -1, :]))
        return out  # logits (no sigmoid)

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_features = batch_features.unsqueeze(1)
            logits = model(batch_features)
            outputs = torch.sigmoid(logits)
            predictions = (outputs > 0.5).float()
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    results = metrics.compute_all(y_true, y_pred)
    print("Metrics:", results)
    return results


def train_model(config, train_loader, val_loader_fn, input_size, return_best_f1=False):
    print(f"[Debug] 🔧 Entered train_model() with config: {config}", flush=True)

    model = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[Debug] 🚀 Model instantiated and moved to device", flush=True)

    # === Step 1: Dynamically compute pos_weight ===
    num_pos, num_neg = 0, 0
    for _, y in train_loader():  # if train_loader is generator, no () needed
        num_pos += (y == 1).sum().item()
        num_neg += (y == 0).sum().item()
    ratio = num_neg / max(1, num_pos)
    print(f"[Info] pos_weight ratio={ratio:.1f}", flush=True)

    pos_weight = torch.tensor([ratio], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print("[Debug] ✅ Optimizer and loss set up", flush=True)

    best_f1 = 0
    epochs = config.get("epochs", 3)
    early_stop_patience = config.get("early_stop_patience", 2)
    patience_counter = 0

    print(f"[Debug] 🗓️ Starting loop for {epochs} epochs", flush=True)

    for epoch in range(epochs):
        model.train()
        print(f"[Debug] 🏁 Epoch {epoch+1}/{epochs} started", flush=True)

        for batch_num, (batch_features, batch_labels) in enumerate(train_loader(), 1):
            if batch_num % 1000 == 0:
                print(f"[Debug] ⛏️  Training batch {batch_num}", flush=True)

            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.float().to(device)
            if batch_labels.dim() == 1:
                batch_labels = batch_labels.unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"[GRU] [Epoch {epoch+1}] ✅ Done.", flush=True)

        if return_best_f1:
            print("[Eval] 🧪 Evaluating F1 for early stopping...", flush=True)
            model.eval()

            metrics_dict = quick_f1(model, val_loader_fn, device)
            precision  = metrics_dict["Precision"]
            recall     = metrics_dict["Recall"]
            f1_score_  = metrics_dict["F1"]
            th         = metrics_dict["th"]

            print(f"[Eval] th={th:.2f}  F1={f1_score_:.4f} | P={precision:.4f} | R={recall:.4f}")


            if f1_score_ > best_f1:
                best_f1 = f1_score_
                patience_counter = 0
                print("🎯 New best F1 found!", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("🛑 Early stopping triggered.", flush=True)
                    break


    print("[Debug] 🎉 Finished training function", flush=True)
    return best_f1 if return_best_f1 else model


