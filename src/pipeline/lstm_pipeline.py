from models.lstm_rnn import train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_chunked
from utils.evaluator import evaluate_and_export
from utils.tuning import manual_gru_search
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
import torch


def run_lstm_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_chunked(redteam_file='data/redteam.txt.gz')
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    # ✅ Load chunked dataset with sequence support
    chunk_dataset = SequenceChunkedDataset(
        chunk_dir=CHUNKS_LABELED_PATH,
        label_column='label',
        batch_size=256,
        shuffle_files=True,
        binary_labels=True,
        sequence_length=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    input_size = chunk_dataset.input_size

    # Step 1: Manual param search
    param_grid = [
        {"lr": 0.001, "hidden_size": 64, "num_layers": 1, "epochs": 6, "early_stop_patience": 2},
        {"lr": 0.0005, "hidden_size": 128, "num_layers": 2, "epochs": 6, "early_stop_patience": 2},
        {"lr": 0.001, "hidden_size": 128, "num_layers": 1, "epochs": 6, "early_stop_patience": 2}
    ]


    def train_func(config):
        return train_model(
            config=config,
            train_loader=chunk_dataset.train_loader,
            val_loader=chunk_dataset.val_loader,
            input_size=input_size,
            return_best_f1=True
        )

    best_config = manual_gru_search(train_func, param_grid)
    if best_config is None:
        print("❌ No valid configuration found. Exiting pipeline.")
        return

    print(f"[Manual Tune] ✅ Best config: {best_config}")

    # Step 2: Final training and export
    model = train_model(
        config=best_config,
        train_loader=chunk_dataset.train_loader,
        val_loader=chunk_dataset.val_loader,
        input_size=input_size,
        return_best_f1=False
    )

    # Step 3: Evaluation
    evaluate_and_export(
        model,
        chunk_dataset.full_loader(),
        model_name="lstm",
        device=chunk_dataset.device,
        export_ground_truth=True
    )