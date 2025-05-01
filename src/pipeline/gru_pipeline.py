from utils.tuning import manual_gru_search
from models.gru import GRUAnomalyDetector, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_chunked
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from utils.chunked_dataset import ChunkedCSVDataset
from utils.constants import CHUNKS_LABELED_PATH
from glob import glob
from utils.metrics import Metrics
import pandas as pd
import torch
import os


def run_gru_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_chunked(redteam_file='data/redteam.txt.gz')
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    chunk_dir = CHUNKS_LABELED_PATH

    # ✅ Step 1: Infer expected features from the first chunk
    first_chunk_path = glob(os.path.join(chunk_dir, "*.csv"))[0]
    df_sample = pd.read_csv(first_chunk_path)
    expected_features = df_sample.drop(columns=['label']).select_dtypes(include=['number']).columns.tolist()
    input_size = len(expected_features)

    # ✅ Step 2: Create the chunked dataset with aligned features
    chunk_dataset = ChunkedCSVDataset(
        chunk_dir=chunk_dir,
        chunk_size=5000,
        label_col='label',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        expected_features=expected_features
    )

    # ✅ Step 3: Define manual hyperparameter grid
    param_grid = [
        {"lr": 0.001, "hidden_size": 64, "num_layers": 1},
        {"lr": 0.0005, "hidden_size": 128, "num_layers": 2},
        {"lr": 0.001, "hidden_size": 128, "num_layers": 1}
    ]

    # ✅ Step 4: Wrap the training function
    def train_func(config):
        return train_model(
            config=config,
            train_loader=chunk_dataset.train_loader(),
            val_loader=chunk_dataset.val_loader(),
            input_size=input_size,
            return_best_f1=True
            )

    # ✅ Step 5: Search for best config manually
    best_config = manual_gru_search(train_func, param_grid)
    print(f"[Manual Tune] ✅ Best hyperparameters: {best_config}")

    # ✅ Step 6: Final training using best config
    model = train_model(
        config=best_config,
        train_loader=chunk_dataset.train_loader(),
        val_loader=chunk_dataset.val_loader(),
        input_size=input_size,
        return_best_f1=False
    )

    # ✅ Step 7: Export & Evaluate
    export_model(model, "/app/models/gru_trained_model.pth")
    evaluate_and_export(
        model,
        chunk_dataset.full_loader(),
        model_name="gru",
        device=chunk_dataset.device,
        export_ground_truth=True
    )

