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

    chunk_dataset = SequenceChunkedDataset(
        chunk_dir=CHUNKS_LABELED_PATH,
        label_column='label',
        batch_size=128,
        shuffle_files=True,
        binary_labels=True,
        sequence_length=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    input_size = chunk_dataset.input_size

    # ✅ Step 1: Manual tuning
    param_grid = [
        {"lr": 0.001, "d_model": 64, "nhead": 2, "num_encoder_layers": 2, "dim_feedforward": 128, "dropout": 0.1}
    ]

    def train_func(config):
        return train_transformer(
            config=config,
            train_loader=chunk_dataset.train_loader,
            val_loader=chunk_dataset.val_loader,
            input_size=input_size,
            return_best_f1=True
        )

    best_config = manual_gru_search(train_func, param_grid)
    print(f"[Manual Tune] ✅ Best config: {best_config}")

    # ✅ Step 2: Final training and export
    model = train_transformer(
        config=best_config,
        train_loader=chunk_dataset.train_loader,
        val_loader=chunk_dataset.val_loader,
        input_size=input_size,
        return_best_f1=False
    )

    # ✅ Step 3: Final evaluation
    evaluate_and_export(
        model,
        chunk_dataset.full_loader(),
        model_name="transformer",
        device=chunk_dataset.device,
        export_ground_truth=True
    )

