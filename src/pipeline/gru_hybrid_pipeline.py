# ─── src/pipeline/gru_hybrid_pipeline.py ────────────────────────────
from glob import glob
import os, pandas as pd, torch
from torch.utils.data import DataLoader, random_split

from utils.balanced_dataset import GlobalBalancedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning import manual_gru_search
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.gru_hybrid import train_gru, train_hybrid

# --------------------------------------------------------------------
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

    # ── 1. Build a global balanced dataset (seq_len = 1 for GRU) ----
    bal_ds = GlobalBalancedDataset(
        chunk_dir,
        feature_cols=list(numeric_cols),
        label_col="label",
        sequence_length=1,
        minority_factor=0.30,          # ≈30 % positives each batch
        device=device,
    )

    # Stratified split: 90 % train / 10 % val  (≥50 anomalies in val)
    val_len   = int(0.10 * len(bal_ds))
    train_len = len(bal_ds) - val_len
    train_set, val_set = random_split(bal_ds, [train_len, val_len])

    while bal_ds.labels[val_set.indices].sum() < 50:          # ensure signal
        train_set, val_set = random_split(bal_ds, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        sampler=bal_ds.sampler,      # weighted sampler → balanced batches
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # helpers compatible with existing train_* functions
    def train_iter():
        yield from train_loader

    def val_once():
        yield from val_loader        # one full pass each call

    # quick stats
    pos_tr  = sum((y == 1).sum().item() for _, y in train_iter())
    pos_val = sum((y == 1).sum().item() for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ----------------------------------------------------------------
    # 2. Grid search for backbone hyper-params
    # ----------------------------------------------------------------
    grid = [
        {"lr":1e-3, "hidden_size":48, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3, "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":48, "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def _train(cfg):
        best_f1, _ = train_gru(
            cfg,
            loaders=(train_iter, val_once),
            input_size=input_size,
            tag=f"gru_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every_epoch=True,
        )
        return best_f1

    best_cfg = manual_gru_search(_train, grid)
    print("🏅 Best GRU config:", best_cfg)

    # ----------------------------------------------------------------
    # 3. Final backbone training with best config
    # ----------------------------------------------------------------
    tag = f"gru_h{best_cfg['hidden_size']}_l{best_cfg['num_layers']}"
    best_f1, gru_model = train_gru(
        best_cfg,
        loaders=(train_iter, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every_epoch=False,
    )
    print(f"👍 Final GRU F1 = {best_f1:.4f}")

    # combined loader for evaluation/export
    def full_loader():
        yield from train_loader
        yield from val_loader

    export_model(gru_model, "/app/models/gru_trained_model.pth")
    evaluate_and_export(
        gru_model,
        full_loader(),
        model_name="gru",
        device=device,
        export_ground_truth=True,
    )

    # ----------------------------------------------------------------
    # 4. Hybrid fine-tune
    # ----------------------------------------------------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid_model = train_hybrid(
        "/app/models/gru_trained_model.pth",
        loaders=(train_iter, val_once),
        input_size=input_size,
        hidden_size=best_cfg["hidden_size"],
        num_layers=best_cfg["num_layers"],
        epochs=3,
        lr=1e-3,
    )
    evaluate_and_export(
        hybrid_model,
        full_loader(),
        model_name="gru_hybrid",
        device=device,
    )

# --------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
