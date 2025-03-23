from utils.tuning import RayTuner
from models.gru import GRUAnomalyDetector, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from torch.utils.data import DataLoader
import pandas as pd
import torch
import ray
from ray import tune
import os


def run_gru_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_with_matching_parallel(
            auth_file='data/auth.txt.gz',
            redteam_file='data/redteam.txt.gz'
        )
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    # Load preprocessed labeled data
    labeled_data_path = 'data/labeled_data/labeled_auth.csv'
    df = pd.read_csv(labeled_data_path)
    print(f"[Pipeline] Loaded dataset shape: {df.shape}")

    # Data Preparation
    labels = df['label'].values
    features = df.drop(columns=['label']).values
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor((labels != -1).astype(float), dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    # DEV environment:
    # train_loader = DataLoader(train_dataset, batch_size=64)
    # val_loader = DataLoader(val_dataset, batch_size=64)

    # Runpod environment:
    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    input_size = features.shape[1]

    # Step 2: Define Ray Tune search space
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3])
    }

    # Step 3: Ray Tune wrapper
    def train_func(config):
        return train_model(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            input_size=input_size
        )

    # Step 4: Run optimization
    tuner = RayTuner(train_func, param_space, num_samples=10, max_epochs=10)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    # Step 5: Train final model with best config
    model = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCELoss()

    # Train manually outside Ray Tune now
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

    # Step 6: Export the model weights
    export_model(model, "/app/models/gru_trained_model.pth")

    # Step 7: Run inference & export predictions
    evaluate_and_export(model, dataset, model_name="gru", device=device)

