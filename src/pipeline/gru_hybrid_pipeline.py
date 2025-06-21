# ---------------------------------------------------------------
#  GRU + MLP hybrid training pipeline (streaming per-chunk dataset)
# ---------------------------------------------------------------
from glob import glob
import os, pandas as pd, torch
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning   import manual_gru_search
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.gru_hybrid import train_gru, train_hybrid

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

    # ---------- dataset (streaming, balanced per chunk) ----------
    dataset = SequenceChunkedDataset(
        chunk_dir,
        label_column="label",
        batch_size=64,
        shuffle_files=True,
        binary_labels=True,
        sequence_length=1,   # GRU uses single-row sequences
        device=device,
        split_ratio=0.9      # 90 % train • 10 % val **per chunk**
    )

    train_iter = dataset.train_loader      # generator
    val_once   = dataset.val_loader        # call ⇒ one full val pass
    full_iter  = dataset.full_loader       # train + val

    # tiny stats for the user
    pos_tr  = sum((y == 1).sum().item() for _, y in train_iter())
    pos_val = sum((y == 1).sum().item() for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ---------- stage-1 : manual grid search ---------------------
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

    # ---------- stage-1b : final backbone training ---------------
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

    # ---------- stage-2 : hybrid fine-tune -----------------------
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

if __name__ == "__main__":
    run_pipeline()
