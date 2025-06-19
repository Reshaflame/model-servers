# src/pipeline/lstm_hybrid_pipeline.py
from glob import glob
import os, pandas as pd, torch
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning import manual_gru_search          # re-use same helper
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.lstm_hybrid import train_lstm, train_hybrid

def run_pipeline():
    chunk_dir = CHUNKS_LABELED_PATH
    first = glob(os.path.join(chunk_dir, "*.csv"))[0]
    numeric = pd.read_csv(first).drop(columns=['label']).select_dtypes('number').columns
    input_size = len(numeric)

    dataset = SequenceChunkedDataset(
        chunk_dir,
        label_column='label',
        batch_size=64, shuffle_files=True, binary_labels=True,
        sequence_length=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    def val_once(): yield from dataset.val_loader()

    # ---- Stage-1: base LSTM+RNN search ---------------------------------------
    grid = [
        {"lr":1e-3, "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":128,"num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3, "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]
    def _train(cfg):
        best_f1, _ = train_lstm(
            cfg,
            loaders=(dataset.train_loader, val_once),
            input_size=input_size,
            tag=f"lstm_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True, eval_every=True
        )
        return best_f1

    best = manual_gru_search(_train, grid)
    print("🏅 Best LSTM config:", best)

    # ---- Final base training (fresh) ----------------------------------------
    tag = f"lstm_h{best['hidden_size']}_l{best['num_layers']}"
    best_f1, lstm_model = train_lstm(
        best,
        loaders=(dataset.train_loader, val_once),
        input_size=input_size,
        tag=tag,
        resume=False, eval_every=False
    )
    print(f"👍 Final base F1 = {best_f1:.4f}")

    export_model(lstm_model, "/app/models/lstm_rnn_trained_model.pth")
    evaluate_and_export(lstm_model, dataset.full_loader(),
                        model_name="lstm", device=dataset.device,
                        export_ground_truth=True)

    # ---- Stage-2: hybrid fine-tune ------------------------------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid("/app/models/lstm_rnn_trained_model.pth",
                          loaders=(dataset.train_loader, val_once),
                          epochs=3, lr=1e-3)
    evaluate_and_export(hybrid, dataset.full_loader(),
                        model_name="lstm_hybrid", device=dataset.device)

if __name__ == "__main__":
    run_pipeline()
