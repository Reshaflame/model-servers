# src/pipeline/tst_pipeline.py

from utils.tuning import RayTuner
from models.transformer import train_transformer, prepare_dataset
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from torch.utils.data import DataLoader
from ray import tune
import ray

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

    # Prepare dataset
    labeled_data_path = 'data/labeled_data/labeled_auth_sample.csv'
    train_dataset, val_dataset = prepare_dataset(labeled_data_path, sequence_length=10)

    # Create DataLoaders
    batch_size = 32  # keep this low for local testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Put heavy objects into Ray Object Store
    train_loader_ref = ray.put(train_loader)
    val_loader_ref = ray.put(val_loader)
    input_size_ref = ray.put(train_dataset[0][0].shape[1])

    # Define search space
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "d_model": tune.choice([32, 64, 128]),
        "nhead": tune.choice([2, 4, 8]),
        "num_encoder_layers": tune.choice([2, 3]),
        "dim_feedforward": tune.choice([128, 256, 512]),
        "dropout": tune.uniform(0.1, 0.3)
    }

    # Ray Tune wrapper
    def train_func(config):
        train_loader_local = ray.get(train_loader_ref)
        val_loader_local = ray.get(val_loader_ref)
        input_size_local = ray.get(input_size_ref)
        return train_transformer(
            config=config,
            train_loader=train_loader_local,
            val_loader=val_loader_local,
            input_size=input_size_local
        )

    # Run Ray Tune
    tuner = RayTuner(train_func, param_space, num_samples=3, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")
