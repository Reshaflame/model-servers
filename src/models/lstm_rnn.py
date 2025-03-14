import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
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

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    device = GPUUtils.get_device()
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'

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
