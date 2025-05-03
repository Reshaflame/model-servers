# src/models/lstm_rnn.py

import torch
import torch.nn as nn
from ray import train as ray_train
import numpy as np
from utils.model_exporter import export_model
from utils.metrics import Metrics


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

    for epoch in range(5):
        model.train()
        print(f"[LSTM] [Epoch {epoch+1}/3] 🔁 Training started...")
        batch_num = 0
        for features, labels in train_loader:
            batch_num += 1
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_num % 1000 == 0:
                print(f"[LSTM]   └─ Batch {batch_num}: Loss = {loss.item():.6f}")

        print(f"[LSTM] [Epoch {epoch+1}] ✅ Done.")

    model.eval()
    metrics = Metrics()
    y_true, y_pred = [], []
    val_loss = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, labels).item()
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    val_loss /= len(val_loader)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics_dict = metrics.compute_standard_metrics(y_true, y_pred)
    metrics_dict["val_loss"] = val_loss

    if return_best_f1:
        return metrics_dict["F1"]
    else:
        export_model(model, "/app/models/lstm_rnn_trained_model.pth")
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
    print("Metrics:", results)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Evaluation Accuracy: {accuracy:.4f}")
