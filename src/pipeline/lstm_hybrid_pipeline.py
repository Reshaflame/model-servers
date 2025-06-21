# ─── src/pipeline/lstm_hybrid_pipeline.py ──────────────────────────
from glob import glob
import os, pandas as pd, torch
from torch.utils.data import DataLoader, random_split

from utils.balanced_dataset import GlobalBalancedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning import manual_gru_search
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.lstm_hybrid import train_lstm, train_hybrid

# ------------------------------------------------------------------
def run_pipeline():
    chunk_dir = CHUNKS_LABELED_PATH
    first     = glob(os.path.join(chunk_dir, "*.csv"))[0]

    numeric_cols = (
        pd.read_csv(first)
          .drop(columns=["label"])
          .select_dtypes("number")
          .columns
    )
    input_size = len(numeric_cols)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. build balanced dataset ----------------------------------
    bal_ds = GlobalBalancedDataset(
        chunk_dir,
        feature_cols=list(numeric_cols),
        label_col="label",
        sequence_length=10,      # LSTM uses 10-step sequences
        minority_factor=0.30,    # ~30 % positives / batch
        device=device,
    )

    # single stratified split: 90 % train / 10 % val
    val_len   = int(0.10 * len(bal_ds))
    train_len = len(bal_ds) - val_len
    train_set, val_set = random_split(bal_ds, [train_len, val_len])

    while bal_ds.labels[val_set.indices].sum() < 50:          # ≥50 anomalies in val
        train_set, val_set = random_split(bal_ds, [train_len, val_len])

    # ── build sampler only if subset is under 16 M rows ─────────────
    y_train = bal_ds.labels[train_set.indices]
    n_pos   = int(y_train.sum()); n_tot = len(y_train)

    if n_tot <= 16_000_000:                      # safe for torch.multinomial
        import numpy as np
        w_pos = 0.30 / max(1, n_pos)
        w_neg = 0.70 / (n_tot - n_pos)
        weights = np.where(y_train == 1, w_pos, w_neg)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights,
                                        num_samples=n_tot,
                                        replacement=True)
        shuffle_flag = False
    else:                                        # fall back to plain shuffle
        sampler = None
        shuffle_flag = True
        print(f"[Sampler] ⚠️  Train set too large ({n_tot:,}); "
              "using shuffled loader + pos_weight in loss.")

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        sampler=bal_ds.sampler,          # may be None
        shuffle=bal_ds.sampler is None,  # shuffle only when sampler absent
        num_workers=min(os.cpu_count(), 8),
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # helpers (lazy generators) so existing trainers stay unchanged
    def train_iter():
        yield from train_loader

    def val_once():
        yield from val_loader     # one full pass each call

    # small stats printout
    pos_tr  = sum((y == 1).sum().item() for _, y in train_iter())
    pos_val = sum((y == 1).sum().item() for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ------------------------------------------------------------------
    # 2. Stage-1 grid search  (backbone)
    # ------------------------------------------------------------------
    grid = [
        {"lr":1e-3, "hidden_size":64,  "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":128, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3, "hidden_size":64,  "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def _train(cfg):
        best_f1, _ = train_lstm(
            cfg,
            loaders=(train_iter, val_once),
            input_size=input_size,
            tag=f"lstm_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every=True,
            label_array=bal_ds.labels
        )
        return best_f1

    best = manual_gru_search(_train, grid)
    print("🏅 Best LSTM config:", best)

    # ------------------------------------------------------------------
    # 3. Final backbone training with best config
    # ------------------------------------------------------------------
    tag = f"lstm_h{best['hidden_size']}_l{best['num_layers']}"
    best_f1, lstm_model = train_lstm(
        best,
        loaders=(train_iter, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every=False,
    )
    print(f"👍 Final backbone F1 = {best_f1:.4f}")

    # combined loader for full-dataset evaluation / export
    def full_loader():
        yield from train_loader
        yield from val_loader

    export_model(lstm_model, "/app/models/lstm_rnn_trained_model.pth")
    evaluate_and_export(
        lstm_model,
        full_loader(),
        model_name="lstm",
        device=device,
        export_ground_truth=True,
    )

    # ------------------------------------------------------------------
    # 4. Hybrid fine-tune
    # ------------------------------------------------------------------
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
        hybrid,
        full_loader(),
        model_name="lstm_hybrid",
        device=device,
    )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
