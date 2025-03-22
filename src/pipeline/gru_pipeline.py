# src/pipeline/gru_pipeline.py

from utils.tuning import RayTuner
from models.gru import GRUAnomalyDetector, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from torch.utils.data import DataLoader
import pandas as pd
import torch
import ray
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

    # Load preprocessed labeled data
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'
    df = pd.read_csv(labeled_data_path)
    
    # Data Preparation
    labels = df['label'].values
    features = df.drop(columns=['label']).values
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor((labels != -1).astype(float), dtype=torch.float32).unsqueeze(1)

    # Split
    dataset = torch.utils.data.TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    train_loader_ref = ray.put(train_loader)
    val_loader_ref = ray.put(val_loader)
    input_size_ref = ray.put(features.shape[1])

    # Step 2: Define Ray Tune search space
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3])
    }

    # Step 3: Ray Tune wrapper
    def train_func(config):
        train_loader_local = ray.get(train_loader_ref)
        val_loader_local = ray.get(val_loader_ref)
        input_size_local = ray.get(input_size_ref)

        return train_model(
            config=config,
            train_loader=train_loader_local,
            val_loader=val_loader_local,
            input_size=input_size_local
        )


    # Step 4: Run optimization
    tuner = RayTuner(train_func, param_space, num_samples=10, max_epochs=10)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")
