# ---------------------------------------------------------------
#  GRU + MLP hybrid training pipeline  (FAST balanced dataset)
# ---------------------------------------------------------------
from glob import glob
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from utils.fast_balanced_dataset import FastBalancedDS
from utils.dl_helpers          import to_device
from utils.constants           import CHUNKS_LABELED_PATH
from utils.tuning              import manual_gru_search
from utils.evaluator           import evaluate_and_export
from utils.model_exporter      import export_model
from models.gru_hybrid         import train_gru, train_hybrid

# ------------------------------------------------------------------
def run_pipeline() -> None:
    # ── dataset meta ------------------------------------------------
    chunk_dir  = CHUNKS_LABELED_PATH
    first_csv  = glob(os.path.join(chunk_dir, "*.csv"))[0]
    numeric    = (
        pd.read_csv(first_csv)
          .drop(columns=["label"])
          .select_dtypes("number")
          .columns
    )
    input_size = len(numeric)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # ── build fast balanced dataset  -------------------------------
    full_ds = FastBalancedDS(
        chunk_dir,
        bank_pt        = "/data/anomaly_bank.pt",   # <── use the correct kw-arg
        feature_cols   = list(numeric),
        seq_len        = 1,                         # GRU uses single-row “sequences”
        minority_ratio = 0.30                       # ≈30 % positives per mini-batch
    )

    # single stratified split (90 % train / 10 % val)
    val_len   = int(0.10 * len(full_ds))
    train_len = len(full_ds) - val_len
    train_set, val_set = random_split(full_ds, [train_len, val_len])

    # ── dataloaders -------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size   = 64,
        shuffle      = True,
        num_workers  = 4,
        pin_memory   = (device != "cpu"),
        collate_fn   = lambda b: to_device(b, device)
    )
    val_loader = DataLoader(
        val_set,
        batch_size   = 64,
        shuffle      = False,
        num_workers  = 2,
        pin_memory   = (device != "cpu"),
        collate_fn   = lambda b: to_device(b, device)
    )

    # adapt existing helpers so the rest of the code is unchanged
    def train_iter():     yield from train_loader
    def val_once():       yield from val_loader
    def full_iter():      yield from train_loader; yield from val_loader

    # quick stats
    pos_tr  = sum(int(y.sum()) for _, y in train_iter())
    pos_val = sum(int(y.sum()) for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ── stage-1 : manual grid search -------------------------------
    grid = [
        {"lr":1e-3 , "hidden_size":48, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3 , "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":48, "num_layers":2, "epochs":6, "early_stop_patience":2},
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

    # ── stage-1b : backbone re-train with best params --------------
    tag = f"gru_h{best_cfg['hidden_size']}_l{best_cfg['num_layers']}"
    best_f1, gru_backbone = train_gru(
        best_cfg,
        loaders=(train_iter, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every_epoch=False,
    )
    print(f"👍 Final GRU F1 on val = {best_f1:.4f}")

    export_model(gru_backbone, "/app/models/gru_trained_model.pth")
    evaluate_and_export(
        gru_backbone, full_iter(), model_name="gru", device=device, export_ground_truth=True
    )

    # ── stage-2 : hybrid fine-tune ---------------------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid(
        "/app/models/gru_trained_model.pth",
        loaders=(train_iter, val_once),
        input_size=input_size,
        hidden_size=best_cfg["hidden_size"],
        num_layers=best_cfg["num_layers"],
        epochs=3,
        lr=1e-3,
    )
    evaluate_and_export(
        hybrid, full_iter(), model_name="gru_hybrid", device=device
    )

# --------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
