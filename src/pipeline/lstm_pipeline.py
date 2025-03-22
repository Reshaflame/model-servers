# src/pipeline/lstm_pipeline.py

from utils.tuning import RayTuner
from models.lstm_rnn import LSTM_RNN_Hybrid, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from torch.utils.data import DataLoader
from ray import tune
import pandas as pd
import torch
import ray

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

    # Load preprocessed labeled data
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'
    df = pd.read_csv(labeled_data_path)
    
    # Data Preparation
    labels = df['label'].values
    features = df.drop(columns=['label']).values
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor((labels != -1).astype(float), dtype=torch.float32).unsqueeze(1)

    # Group into sequences
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

    # Put large objects into Ray object store
    train_loader_ref = ray.put(train_loader)
    val_loader_ref = ray.put(val_loader)
    input_size_ref = ray.put(sequences.shape[2])

    # Step 2: Define Ray Tune search space
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2])
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
    tuner = RayTuner(train_func, param_space, num_samples=4, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")
