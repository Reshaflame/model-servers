# src/pipeline/lstm_pipeline.py

from utils.tuning import RayTuner
from models.lstm_rnn import LSTM_RNN_Hybrid, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from utils.model_exporter import export_model
from torch.utils.data import DataLoader
from ray import tune
import pandas as pd
import torch

def run_lstm_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_with_matching_parallel(
            auth_file='data/auth.txt.gz',
            redteam_file='data/redteam.txt.gz'
        )
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'
    df = pd.read_csv(labeled_data_path)

    labels = df['label'].values
    features = df.drop(columns=['label']).values
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor((labels != -1).astype(float), dtype=torch.float32).unsqueeze(1)

    sequence_length = 10
    sequences, sequence_labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i+sequence_length])
        sequence_labels.append(labels[i+sequence_length-1])
    sequences = torch.stack(sequences)
    sequence_labels = torch.stack(sequence_labels)

    dataset = torch.utils.data.TensorDataset(sequences, sequence_labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)

    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2])
    }

    def train_func(config):
        return train_model(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            input_size=features.shape[1]
        )

    tuner = RayTuner(train_func, param_space, num_samples=4, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    model = LSTM_RNN_Hybrid(
        input_size=features.shape[1],
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCELoss()

    for epoch in range(5):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    export_model(model, "/app/models/lstm_rnn_trained_model.pth")
