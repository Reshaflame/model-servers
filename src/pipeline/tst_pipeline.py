# src/pipeline/tst_pipeline.py

from utils.tuning import RayTuner
from models.transformer import TimeSeriesTransformer, train_transformer
from preprocess.labeledPreprocess import preprocess_labeled_data_chunked
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.model_exporter import export_model
from utils.evaluator import evaluate_and_export
from utils.constants import CHUNKS_LABELED_PATH
from ray import tune
import torch


def run_tst_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_chunked(redteam_file='data/redteam.txt.gz')
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    # ✅ Initialize sequence-aware chunked dataset loader
    chunk_dir = CHUNKS_LABELED_PATH
    sequence_chunks = SequenceChunkedDataset(
        chunk_dir=chunk_dir,
        sequence_length=10,
        label_column='label',
        batch_size=128,
        shuffle_files=True,
        binary_labels=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    input_size = sequence_chunks.input_size

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
            train_loader=sequence_chunks.train_loader(),
            val_loader=sequence_chunks.val_loader(),
            input_size=input_size
        )

    tuner = RayTuner(train_func, param_space, num_samples=3, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    device = sequence_chunks.device
    model = TimeSeriesTransformer(
        input_size=input_size,
        d_model=best_config["d_model"],
        nhead=best_config["nhead"],
        num_encoder_layers=best_config["num_encoder_layers"],
        dim_feedforward=best_config["dim_feedforward"],
        dropout=best_config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        for batch_features, batch_labels in sequence_chunks.train_loader():
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    export_model(model, "/app/models/transformer_trained_model.pth")

    evaluate_and_export(
        model,
        sequence_chunks.full_loader(),
        model_name="transformer",
        device=device,
        export_ground_truth=True
    )
