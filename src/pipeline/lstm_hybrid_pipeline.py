# ---------------------------------------------------------------
#  LSTM+RNN + MLP hybrid training pipeline (streaming dataset)
# ---------------------------------------------------------------
from glob import glob
import os, pandas as pd, torch
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning   import manual_gru_search          # reuse same helper
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.lstm_hybrid  import train_lstm, train_hybrid

def run_pipeline() -> None:
    chunk_dir = CHUNKS_LABELED_PATH
    first_csv = glob(os.path.join(chunk_dir, "*.csv"))[0]
    numeric   = (
        pd.read_csv(first_csv)
          .drop(columns=["label"])
          .select_dtypes("number")
          .columns
    )
    input_size = len(numeric)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- dataset -----------------------------------------
    dataset = SequenceChunkedDataset(
        chunk_dir,
        label_column="label",
        batch_size=64,
        shuffle_files=True,
        binary_labels=True,
        sequence_length=10,   # 10-step sequences
        device=device,
        split_ratio=0.9
    )

    train_iter = dataset.train_loader
    val_once   = dataset.val_loader
    full_iter  = dataset.full_loader

    pos_tr  = sum((y == 1).sum().item() for _, y in train_iter())
    pos_val = sum((y == 1).sum().item() for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ---------- stage-1 : grid search ---------------------------
    grid = [
        {"lr":1e-3 , "hidden_size":64 , "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":128, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3 , "hidden_size":64 , "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def _train(cfg):
        best_f1, _ = train_lstm(
            cfg,
            loaders=(train_iter, val_once),
            input_size=input_size,
            tag=f"lstm_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every=True,
        )
        return best_f1

    best = manual_gru_search(_train, grid)
    print("🏅 Best LSTM config:", best)

    # ---------- stage-1b : backbone train -----------------------
    tag = f"lstm_h{best['hidden_size']}_l{best['num_layers']}"
    best_f1, lstm_backbone = train_lstm(
        best,
        loaders=(train_iter, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every=False,
    )
    print(f"👍 Final backbone F1 = {best_f1:.4f}")

    export_model(lstm_backbone, "/app/models/lstm_rnn_trained_model.pth")
    evaluate_and_export(
        lstm_backbone, full_iter(), model_name="lstm",
        device=device, export_ground_truth=True
    )

    # ---------- stage-2 : hybrid fine-tune ----------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid(
        "/app/models/lstm_rnn_trained_model.pth",
        loaders=(train_iter, val_once),
        input_size=input_size,
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        epochs=3,
        lr=1e-3,
    )
    evaluate_and_export(
        hybrid, full_iter(), model_name="lstm_hybrid", device=device
    )

if __name__ == "__main__":
    run_pipeline()
