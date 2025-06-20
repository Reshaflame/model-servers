# ─── src/pipeline/lstm_hybrid_pipeline.py ─────────────────────────────
from glob import glob
import os, pandas as pd, torch
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning import manual_gru_search
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.lstm_hybrid import train_lstm, train_hybrid

def run_pipeline():
    chunk_dir = CHUNKS_LABELED_PATH
    first = glob(os.path.join(chunk_dir, "*.csv"))[0]
    numeric = (
        pd.read_csv(first)
          .drop(columns=["label"])
          .select_dtypes("number")
          .columns
    )
    input_size = len(numeric)

    # ── Build dataset (will split per-chunk internally) ──────────────
    dataset = SequenceChunkedDataset(
        chunk_dir,
        label_column="label",
        batch_size=64,
        shuffle_files=True,
        split_ratio=0.9,           # 90 % train / 10 % val
        binary_labels=True,
        sequence_length=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ── Ensure val set contains ≥1 anomaly overall ───────────────────
    def _count_pos(loader_fn):
        p = t = 0
        for _, y in loader_fn():
            p += (y == 1).sum().item()
            t += y.numel()
        return p, t

    while True:
        pos_val, tot_val = _count_pos(dataset.val_loader)
        if pos_val > 0:
            break
        print("🔄 Re-seeding split — val had no positives")
        dataset.resplit()

    pos_tr, tot_tr = _count_pos(dataset.train_loader)
    print(f"🧮 train positives ≈ {pos_tr}/{tot_tr}")
    print(f"🧮  val  positives ≈ {pos_val}/{tot_val}")

    def val_once():                 # single pass each call
        yield from dataset.val_loader()

    # ── Stage 1: hyper-param grid search ─────────────────────────────
    grid = [
        {"lr":1e-3, "hidden_size":64,  "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":128, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3, "hidden_size":64,  "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]
    def _train(cfg):
        best_f1, _ = train_lstm(
            cfg,
            loaders=(dataset.train_loader, val_once),
            input_size=input_size,
            tag=f"lstm_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every=True,
        )
        return best_f1

    best = manual_gru_search(_train, grid)
    print("🏅 Best LSTM config:", best)

    # ── Stage 1b: final backbone training ────────────────────────────
    tag = f"lstm_h{best['hidden_size']}_l{best['num_layers']}"
    best_f1, lstm_model = train_lstm(
        best,
        loaders=(dataset.train_loader, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every=False,
    )
    print(f"👍 Final backbone F1 = {best_f1:.4f}")

    export_model(lstm_model, "/app/models/lstm_rnn_trained_model.pth")
    evaluate_and_export(
        lstm_model,
        dataset.full_loader(),
        model_name="lstm",
        device=dataset.device,
        export_ground_truth=True,
    )

    # ── Stage 2: hybrid fine-tune ────────────────────────────────────
    print("\\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid(
        "/app/models/lstm_rnn_trained_model.pth",
        loaders=(dataset.train_loader, val_once),
        input_size=input_size,
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        epochs=3,
        lr=1e-3,
    )
    evaluate_and_export(
        hybrid,
        dataset.full_loader(),
        model_name="lstm_hybrid",
        device=dataset.device,
    )

if __name__ == "__main__":
    run_pipeline()
