# src/pipeline/tst_pipeline.py

from utils.tuning import manual_gru_search
from models.transformer import TimeSeriesTransformer, train_transformer
from preprocess.labeledPreprocess import preprocess_labeled_data_chunked
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.model_exporter import export_model
from utils.evaluator import evaluate_and_export
from utils.constants import CHUNKS_LABELED_PATH
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

    param_grid = {
        "lr": [1e-3],
        "d_model": [64],
        "nhead": [2],
        "num_encoder_layers": [2],
        "dim_feedforward": [128],
        "dropout": [0.1]
    }
    
    def train_func(config):
        return train_transformer(
            config=config,
            train_loader=sequence_chunks.train_loader(),
            val_loader=sequence_chunks.val_loader(),
            input_size=input_size
        )

    best_config = manual_gru_search(train_func, param_grid)
    print(f"[Manual Search] Best hyperparameters: {best_config}")

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
