import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ray import train as ray_train
from utils.metrics import Metrics
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.model_exporter import export_model

metrics = Metrics()

# Define the GRU model
class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRUAnomalyDetector, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, _ = self.gru(x)
        out = self.fc(h[:, -1, :])  # Take the output of the last time step
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

def prepare_dataset(file_path):
    print("Loading preprocessed data...")
    data = pd.read_csv(file_path)
    labels = data['label'].values
    features = data.drop(columns=['label']).values
    labels = np.where(labels == -1, 0, labels)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def train_model(config, train_loader, val_loader, input_size):
    model = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()

    for epoch in range(10):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_features = batch_features.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Export trained model after training loop
    export_model(model, "/app/models/gru_trained_model.pth")

    # Evaluation for Ray Tune
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            preds = model(features.unsqueeze(1))
            loss = criterion(preds, labels)
            val_loss += loss.item()
            preds = (preds > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    val_loss /= len(val_loader)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics_dict = metrics.compute_standard_metrics(y_true, y_pred)
    metrics_dict["val_loss"] = val_loss
    ray_train.report(metrics_dict)

if __name__ == "__main__":
    labeled_data_path = 'data/labeled_data/labeled_auth.csv'
    train_dataset, test_dataset = prepare_dataset(labeled_data_path)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = len(train_dataset[0][0])
    model = GRUAnomalyDetector(input_size).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop for manual tests
    for epoch in range(10):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(model.device), batch_labels.to(model.device)
            batch_features = batch_features.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    export_model(model, "gru_manual_test_model.pth")
    evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
