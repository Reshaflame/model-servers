import numpy as np
import torch
import torch.nn as nn
from utils.metrics import Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.model_exporter import export_model

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

def train_model(config, train_loader, val_loader, input_size, return_best_f1=False):
    model = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    pos_weight = torch.tensor([100.0], device=device)  # ✅ Default weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    epochs = config.get("epochs", 3)
    early_stop_patience = config.get("early_stop_patience", 2)
    patience_counter = 0

    all_val_labels = []
    for _, labels in val_loader():
        all_val_labels.extend(labels.cpu().numpy().flatten())

    unique, counts = np.unique(all_val_labels, return_counts=True)
    print("[Debug] Validation Label Distribution:", dict(zip(unique, counts)))


    for epoch in range(epochs):
        model.train()
        print(f"[GRU] [Epoch {epoch+1}/{epochs}] 🔁 Training started...")
        for batch_num, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            if batch_num % 1000 == 0:
                print(f"[GRU]   └─ Batch {batch_num}: Loss = {loss.item():.6f}")

        print(f"[GRU] [Epoch {epoch+1}] ✅ Done.")

        if return_best_f1:
            print("[Eval] 🧪 Evaluating F1 for early stopping...")
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for features, labels in val_loader():
                    features = features.float().to(device)
                    if features.dim() == 2:
                        features = features.unsqueeze(1)
                    labels = labels.float().to(device)
                    preds = torch.sigmoid(model(features))

                    # 🔍 Optional: log only on first batch to avoid spam
                    if len(y_true) == 0:
                        print("[Eval Debug] Logits:", preds[:5].squeeze().cpu().numpy())
                        print("[Eval Debug] Labels:", labels[:5].cpu().numpy())


            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"[Eval] F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                print("🎯 New best F1 found!")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("🛑 Early stopping triggered.")
                    break

    return best_f1 if return_best_f1 else model
