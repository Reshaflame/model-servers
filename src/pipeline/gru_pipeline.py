from utils.tuning import RayTuner
from models.gru import GRUAnomalyDetector, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from utils.chunked_dataset import ChunkedCSVDataset
from torch.utils.data import DataLoader
import torch
from ray import tune


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

    # ✅ Use chunked dataset instead of loading all data into memory
    chunk_dir = "data/labeled_data/chunks"
    chunk_dataset = ChunkedCSVDataset(
        chunk_dir=chunk_dir,
        label_column='label',
        batch_size=64,
        shuffle_files=True,
        binary_labels=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Get input size from the first chunk
    input_size = chunk_dataset.input_size

    # Step 2: Define Ray Tune search space
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3])
    }

    def train_func(config):
        return train_model(
            config=config,
            train_loader=chunk_dataset.train_loader(),
            val_loader=chunk_dataset.val_loader(),
            input_size=input_size
        )

    tuner = RayTuner(train_func, param_space, num_samples=10, max_epochs=10)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    # Step 5: Train final model
    model = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"]
    ).to(chunk_dataset.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCELoss()

    for epoch in range(10):
        model.train()
        for batch_features, batch_labels in chunk_dataset.train_loader():
            batch_features = batch_features.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    export_model(model, "/app/models/gru_trained_model.pth")

    # ✅ Evaluate using all chunks (y_true/y_pred saved internally)
    evaluate_and_export(
        model,
        chunk_dataset.full_loader(),
        model_name="gru",
        device=chunk_dataset.device,
        export_ground_truth=True
    )
