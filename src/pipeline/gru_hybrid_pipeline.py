# ---------------------------------------------------------------
#  GRU + MLP hybrid training pipeline  (FAST balanced dataset)
# ---------------------------------------------------------------
from glob import glob
import os, pandas as pd, torch, random
from torch.utils.data import DataLoader

from utils.fast_balanced_dataset import FastBalancedDS
from utils.build_anomaly_bank    import build_if_needed
from utils.dl_helpers            import stack_batch
from utils.constants             import CHUNKS_LABELED_PATH
from utils.tuning                import manual_gru_search
from utils.evaluator             import evaluate_and_export
from utils.model_exporter        import export_model
from models.gru_hybrid           import train_gru, train_hybrid

# ---------------------------------------------------------------
def run_pipeline() -> None:
    # ── 0) meta ----------------------------------------------------
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

    # ── 1) anomaly bank (build once, very fast) --------------------
    BANK_PT = "/workspace/model-servers/data/anomaly_bank.pt"
    build_if_needed(
        src_dir      = chunk_dir,
        out_pt       = BANK_PT,
        feature_cols = list(numeric),
        seq_len      = 1                # GRU = single-row sequences
    )
    bank = torch.load(BANK_PT, map_location="cpu")
    n_bank_pos = bank["X"].shape[0]     # positives available for eval

    # ── 2) datasets & loaders --------------------------------------
    POS_RATIO = 0.30

    train_ds = FastBalancedDS(
        chunk_dir      = chunk_dir,
        bank_pt        = BANK_PT,
        feature_cols   = list(numeric),
        seq_len        = 1,
        pos_ratio      = POS_RATIO
    )
    train_loader = DataLoader(
        train_ds,
        batch_size   = 64,
        shuffle      = True,
        num_workers  = 4,
        pin_memory   = (device != "cpu"),
        collate_fn   = stack_batch          # already returns tensors on device
    )

    # --- build a *fixed* validation set: every positive + 4× negatives
    val_X, val_y = [bank["X"]], [bank["y"]]
    neg_needed   = 4 * n_bank_pos
    rng = random.Random(0)   # deterministic
    for _ in range(neg_needed):
        x, y = train_ds.sample_negative()   # helper we added below
        val_X.append(x.unsqueeze(0))
        val_y.append(torch.zeros(1, 1))
    val_X = torch.cat(val_X).to(device)
    val_y = torch.cat(val_y).to(device)

    def val_once():
        yield val_X, val_y

    # quick stats ---------------------------------------------------
    est_neg_batch = int((next(iter(train_loader))[1] == 0).sum())
    print(f"🧮 positives in bank        : {n_bank_pos}")
    print(f"🧮 negatives per train batch: {est_neg_batch}")

    # ── 3) grid search --------------------------------------------
    grid = [
        {"lr":1e-3 , "hidden_size":48, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3 , "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":48, "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def train_loader_fn():
        yield from train_loader          # wrapper to match old API

    def _train(cfg):
        best_f1, _ = train_gru(
            cfg,
            loaders=(train_loader_fn, val_once),
            input_size=input_size,
            tag=f"gru_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every_epoch=True,
            n_bank_pos = n_bank_pos,
            pos_ratio  = POS_RATIO
        )
        return best_f1

    best_cfg = manual_gru_search(_train, grid)
    print("🏅 best GRU config", best_cfg)

    # --- 3b final backbone run ------------------------------------
    tag = f"gru_h{best_cfg['hidden_size']}_l{best_cfg['num_layers']}"
    best_f1, backbone = train_gru(
        best_cfg,
        loaders=(train_loader_fn, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every_epoch=False,
        n_bank_pos = n_bank_pos,
        pos_ratio  = POS_RATIO
    )
    print(f"👍 final val F1 = {best_f1:.4f}")

    export_model(backbone, "/app/models/gru_trained_model.pth")
    evaluate_and_export(backbone, [(val_X, val_y)], "gru", device)

    # ── 4) hybrid fine-tune ---------------------------------------
    print("\n🚀 starting hybrid fine-tune…")
    hybrid = train_hybrid(
        "/app/models/gru_trained_model.pth",
        loaders=(train_loader_fn, val_once),
        input_size=input_size,
        hidden_size=best_cfg["hidden_size"],
        num_layers=best_cfg["num_layers"],
        epochs=3,
        lr=1e-3
    )
    evaluate_and_export(hybrid, [(val_X, val_y)], "gru_hybrid", device)


# small helper added to FastBalancedDS (utils/fast_balanced_dataset.py)
# ---------------------------------------------------------------
# def sample_negative(self):
#     """Return one *negative* sample (X, y) as tensors on CPU."""
#     while True:
#         X, y = super().__getitem__(0)      # forces a random draw
#         if y.item() == 0:
#             return X, y
# ---------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
