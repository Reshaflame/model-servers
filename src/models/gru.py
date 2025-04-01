import numpy as np
import torch
import torch.nn as nn
from ray import train as ray_train
from utils.metrics import Metrics
from sklearn.metrics import precision_score, recall_score, f1_score


from utils.model_exporter import export_model

metrics = Metrics()

class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRUAnomalyDetector, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 🧠 Ensure input is 3D for GRU: [batch, seq_len, input_size]
        if x.dim() == 2:
            x = x.unsqueeze(1)
    
        x = x.contiguous()
        h, _ = self.gru(x)
        out = self.fc(h[:, -1, :])
        return self.sigmoid(out)

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_features = batch_features.unsqueeze(1)
            outputs = model(batch_features)
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
    criterion = nn.BCELoss()

    for epoch in range(3):
        model.train()
        print(f"[Epoch {epoch+1}/3] 🔁 Training started...")
        batch_num = 0
        for batch_features, batch_labels in train_loader:
            batch_num += 1
            batch_features = batch_features.clone().detach().float().to(device)
            batch_labels = batch_labels.clone().detach().float().to(device)
            optimizer.zero_grad()

            # # Before outputs = model(batch_features)
            # print(f"[Train] batch_features shape: {batch_features.shape}")

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.unsqueeze(1))  # 🔥 Match [B, 1] shape
            loss.backward()
            optimizer.step()
            if batch_num % 100 == 0:
                print(f"   └─ Batch {batch_num}: Loss = {loss.item():.6f}")
        print(f"[Epoch {epoch+1}] ✅ Done.")
        
    # === Evaluation after training ===
    print("[Eval] 🔍 Evaluating on validation set...")
    model.eval()
    val_loss = 0
    total_preds = []
    total_labels = []
    batch_id = 0

    with torch.no_grad():
        for features, labels in val_loader:
            batch_id += 1
            features = torch.tensor(features, dtype=torch.float32) if not torch.is_tensor(features) else features.to(dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32) if not torch.is_tensor(labels) else labels.to(dtype=torch.float32)
            features = features.to(device)
            labels = labels.to(device)

            preds = model(features.unsqueeze(1))
            loss = criterion(preds, labels.unsqueeze(1))
            val_loss += loss.item()

            preds_np = (preds > 0.5).float().cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            total_preds.extend(preds_np)
            total_labels.extend(labels_np)

            if batch_id % 100 == 0:
                print(f"   └─ [Eval] Processed {batch_id} batches")

    val_loss /= max(1, batch_id)

    precision = precision_score(total_labels, total_preds, zero_division=0)
    recall = recall_score(total_labels, total_preds, zero_division=0)
    f1 = f1_score(total_labels, total_preds, zero_division=0)

    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "val_loss": val_loss
    }

    print(f"[Eval] ✅ Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {val_loss:.6f}")

    if return_best_f1:
        return metrics_dict["F1"]

    # 🟡 For Ray Tune
    try:
        from ray.train import report as ray_report
        ray_report(metrics_dict)
    except ImportError:
        print("[Warning] Ray not installed or not running in Ray context.")

    return model

