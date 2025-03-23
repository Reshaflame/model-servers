import torch
import os
import torch.nn as nn
from ray import train as ray_train
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.metrics import Metrics
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.gpu_utils import GPUUtils

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

def prepare_dataset(file_path, sequence_length=10):
    print("Loading preprocessed data...")
    data = pd.read_csv(file_path)

    # Separate features and labels
    labels = data['label'].values
    features = data.drop(columns=['label']).values

    # Map labels: -1 -> 0 (anomaly), 1 -> 1 (normal)
    labels = np.where(labels == -1, 0, labels)

    # Group into sequences
    grouped_features, grouped_labels = [], []
    for i in range(len(features) - sequence_length):
        grouped_features.append(features[i:i+sequence_length])
        grouped_labels.append(labels[i+sequence_length-1])

    grouped_features = np.array(grouped_features)
    grouped_labels = np.array(grouped_labels)

    # Convert to tensors
    features_tensor = torch.tensor(grouped_features, dtype=torch.float32)
    labels_tensor = torch.tensor(grouped_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(features_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def train_model(config, train_loader, val_loader, input_size):
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
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation for Ray Tune
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

    # Ray Tune reporting
    ray_train.report(metrics_dict)

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
    results = metrics.compute_all(
        y_true,
        y_pred,
        # anomaly_ranges=your_gt_ranges,  # Optional for TaPR if you can pass intervals
        # pred_ranges=your_predicted_ranges  # Optional
    )
    print("Metrics:", results)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__": # for manual tests
    device = GPUUtils.get_device()
    labeled_data_path = 'data/labeled_data/labeled_auth.csv'

    # Prepare dataset
    train_dataset, test_dataset = prepare_dataset(labeled_data_path)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the model, criterion, and optimizer
    input_size = train_dataset[0][0].shape[1]
    model = LSTM_RNN_Hybrid(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    train_model(model, train_loader, criterion, optimizer, device)
    evaluate_model(model, test_loader, device)
