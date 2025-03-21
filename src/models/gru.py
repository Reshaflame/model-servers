import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils.metrics import Metrics
from torch.utils.data import DataLoader, TensorDataset, random_split

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
    """
    Evaluate the GRU model on the test dataset.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Reshape features to include sequence length (sequence_length = 1)
            batch_features = batch_features.unsqueeze(1)

            # Predict
            outputs = model(batch_features)
            predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions

            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Calculate metrics
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    results = metrics.compute_all(
        y_true,
        y_pred,
        # anomaly_ranges=your_gt_ranges,  # Optional for TaPR if you can pass intervals
        # pred_ranges=your_predicted_ranges  # Optional
    )
    print("Metrics:", results)

    # Optionally, you can add additional metrics like precision, recall, F1-score
    return results


# Prepare the dataset
def prepare_dataset(file_path):
    print("Loading preprocessed data...")
    data = pd.read_csv(file_path)
    
    # Separate features and labels
    labels = data['label'].values
    features = data.drop(columns=['label']).values

    # Map labels: 1 -> 1 (normal), -1 -> 0 (anomalous)
    labels = np.where(labels == -1, 0, labels)

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Create a PyTorch dataset
    dataset = TensorDataset(features, labels)

    # Split into training and testing datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


# Train the GRU model
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Reshape features to include sequence length (sequence_length = 1)
            batch_features = batch_features.unsqueeze(1)  # Add sequence length dimension

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# Evaluate the GRU model
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Reshape features to include sequence length (sequence_length = 1)
            batch_features = batch_features.unsqueeze(1)  # Add sequence length dimension

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# Main function
if __name__ == "__main__":
    # File path to preprocessed data
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'

    # Load and prepare the dataset
    train_dataset, test_dataset = prepare_dataset(labeled_data_path)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model, criterion, and optimizer
    input_size = len(train_dataset[0][0])  # Number of features
    model = GRUAnomalyDetector(input_size).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluate the model
    evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
