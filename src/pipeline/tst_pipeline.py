# src/pipeline/tst_pipeline.py

from utils.tuning import RayTuner
from models.transformer import TimeSeriesTransformer, prepare_dataset, train_transformer
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from utils.model_exporter import export_model
from utils.evaluator import evaluate_and_export
from torch.utils.data import DataLoader
from ray import tune
import torch

def run_tst_pipeline(preprocess=False):
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
    train_dataset, val_dataset = prepare_dataset(labeled_data_path, sequence_length=10)

    # for Dev environment:
    # train_loader = DataLoader(train_dataset, batch_size=32)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    # for Runpod environment:
    train_loader = DataLoader(train_dataset, batch_size=128)
    val_loader = DataLoader(val_dataset, batch_size=128)

    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "d_model": tune.choice([32, 64, 128]),
        "nhead": tune.choice([2, 4, 8]),
        "num_encoder_layers": tune.choice([2, 3]),
        "dim_feedforward": tune.choice([128, 256, 512]),
        "dropout": tune.uniform(0.1, 0.3)
    }

    def train_func(config):
        return train_transformer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            input_size=train_dataset[0][0].shape[1]
        )

    tuner = RayTuner(train_func, param_space, num_samples=3, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    model = TimeSeriesTransformer(
        input_size=train_dataset[0][0].shape[1],
        d_model=best_config["d_model"],
        nhead=best_config["nhead"],
        num_encoder_layers=best_config["num_encoder_layers"],
        dim_feedforward=best_config["dim_feedforward"],
        dropout=best_config["dropout"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(model.device), batch_labels.to(model.device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine train + val into a single dataset for ensemble prediction output
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    export_model(model, "/app/models/transformer_trained_model.pth")

    evaluate_and_export(model, full_dataset, model_name="transformer", device=device)