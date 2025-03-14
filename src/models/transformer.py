import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.gpu_utils import GPUUtils


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Fixed length
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.fc = nn.Linear(d_model, 1)  # No sigmoid here

    def forward(self, x):
        seq_len = x.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :]
        x = self.input_projection(x) + positional_encoding
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])  # Use the last time step
        return torch.sigmoid(x)  # Normalize to [0, 1]



def prepare_dataset(file_path, sequence_length=10):
    print("Loading preprocessed data...")
    data = pd.read_csv(file_path)

    # Oversample anomalies
    anomaly_data = data[data['label'] == -1]
    augmented_anomaly_data = anomaly_data.sample(n=len(anomaly_data) * 10, replace=True, random_state=42)
    balanced_data = pd.concat([data[data['label'] == 1], augmented_anomaly_data])

    # Group into sequences
    grouped_data = []
    for i in range(0, len(balanced_data) - sequence_length, sequence_length):
        sequence = balanced_data.iloc[i:i + sequence_length].drop(columns=['label']).values
        label = balanced_data.iloc[i + sequence_length - 1]['label']
        grouped_data.append((sequence, label))

    # Normalize features
    features = np.array([seq[0] for seq in grouped_data])

    # Normalize features: zero mean, unit variance
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0) + 1e-6  # Avoid division by zero
    features = (features - features_mean) / features_std

    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)  # Normalize to prevent NaNs
    labels = np.array([seq[1] for seq in grouped_data])
    
    print(f"Feature normalization applied: mean={features_mean}, std={features_std}")

    # Convert to tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Create a PyTorch dataset
    dataset = TensorDataset(features, labels)

    # Split into training and testing datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Dataset prepared: {len(train_dataset)} training sequences, {len(test_dataset)} testing sequences")
    return train_dataset, test_dataset


def train_transformer(model, train_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Debug outputs
            print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")
            
            loss = criterion(outputs, batch_labels)
            
            if torch.isnan(loss):
                print("NaN loss encountered!")
                break  # Exit loop on NaN loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Step the scheduler
        if scheduler:
            scheduler.step()



def evaluate_transformer(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).cpu()
            y_pred.extend((torch.sigmoid(outputs) > 0.5).float().numpy())
            y_true.extend(batch_labels.numpy())

    y_true, y_pred = torch.tensor(y_true), torch.tensor(y_pred)
    accuracy = (y_true == y_pred).float().mean().item()
    print(f"Evaluation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'
    device = GPUUtils.get_device()

    # Load and prepare the dataset
    train_dataset, test_dataset = prepare_dataset(labeled_data_path, sequence_length=10)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model, criterion, and optimizer
    input_size = len(train_dataset[0][0][0])  # Number of features
    model = TimeSeriesTransformer(input_size).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Step decay every 5 epochs


    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Transformer):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    model.apply(init_weights)



    # Train the model
    train_transformer(model, train_loader, criterion, optimizer, device, epochs=10, scheduler=scheduler)

    # Evaluate the model
    evaluate_transformer(model, test_loader, device)
