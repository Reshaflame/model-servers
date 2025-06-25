# ---------------------------------------------------------------
#  GRU + MLP hybrid training pipeline  (FAST balanced dataset)
# ---------------------------------------------------------------
from glob import glob
import os, random, torch, pandas as pd
from torch.utils.data import DataLoader

from utils.fast_balanced_dataset import FastBalancedDS                # ⬅ RAM-banks dataset
from utils.build_anomaly_bank    import build_if_needed as build_pos_bank
from utils.build_negative_bank   import build_negative_bank
from utils.dl_helpers            import stack_batch, to_device
from utils.constants             import CHUNKS_LABELED_PATH
from utils.tuning                import manual_gru_search
from utils.evaluator             import evaluate_and_export
from utils.model_exporter        import export_model
from models.gru_hybrid           import train_gru, train_hybrid

# ------------------------------------------------------------------ #
BANK_PT_POS = "/workspace/model-servers/data/anomaly_bank.pt"
BANK_PT_NEG = "/workspace/model-servers/data/negative_bank.pt"
POS_RATIO   = 0.30                 # ≈ 30 % positives per mini-batch
BATCH_SIZE  = 64
NUM_WORKERS = 4                    # ← adjust to your CPU

# ------------------------------------------------------------------ #
def run_pipeline() -> None:
    # ---------- 0. metadata ---------------------------------------
    chunk_dir  = CHUNKS_LABELED_PATH
    first_csv  = glob(os.path.join(chunk_dir, "*.csv"))[0]

    numeric_cols = (
        pd.read_csv(first_csv)
          .drop(columns=["label"])
          .select_dtypes("number")
          .columns
    )
    input_size = len(numeric_cols)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1. build banks (if missing) -----------------------
    build_pos_bank(chunk_dir, BANK_PT_POS,
                   feature_cols=list(numeric_cols), seq_len=1)
    if not os.path.exists(BANK_PT_NEG):
        build_negative_bank()                     # one-off – very fast

    # ---------- 2. dataset & loaders -----------------------------
    full_ds = FastBalancedDS(
        chunk_dir   = chunk_dir,
        bank_pt     = BANK_PT_POS,
        neg_bank_pt = BANK_PT_NEG,
        feature_cols= list(numeric_cols),
        seq_len     = 1,
        pos_ratio   = POS_RATIO,
    )

    train_loader = DataLoader(
        full_ds,
        batch_size   = BATCH_SIZE,
        shuffle      = True,
        num_workers  = NUM_WORKERS,
        pin_memory   = (device != "cpu"),
        collate_fn   = stack_batch        # returns tensors on CPU
    )

    # ---- build a **fixed** validation tensor --------------------
    bank = torch.load(BANK_PT_POS, map_location="cpu")
    n_bank_pos = bank["X"].shape[0]

    val_X = [bank["X"]]                       # list of tensors
    val_y = [bank["y"]]
    neg_needed = 4 * n_bank_pos              # 4× more negatives
    rng = random.Random(0)                   # deterministic

    for _ in range(neg_needed):
        x_cpu, _ = full_ds.sample_negative() # guaranteed label 0
        val_X.append(x_cpu.unsqueeze(0))
        val_y.append(torch.zeros(1, 1))

    val_X = torch.cat(val_X).to(device)
    val_y = torch.cat(val_y).to(device)

    def val_once():
        """One-shot generator expected by train_gru()."""
        yield val_X, val_y

    # ---------- 3. tiny dataset stats ----------------------------
    with torch.no_grad():
        tmp_X, tmp_y = next(iter(train_loader))
        est_neg_batch = int((tmp_y == 0).sum())
    print(f"🧮 positives in bank        : {n_bank_pos}")
    print(f"🧮 negatives per train batch: {est_neg_batch}")

    # ---------- 4. grid search -----------------------------------
    grid = [
        {"lr":1e-3, "hidden_size":48, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3, "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4, "hidden_size":48, "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def train_iter():                           # wrapper — yields CUDA tensors
        for cpu_batch in train_loader:
            yield to_device(cpu_batch, device)

    def _train(cfg):
        best_f1, _ = train_gru(
            cfg,
            loaders        = (train_iter, val_once),
            input_size     = input_size,
            tag            = f"gru_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume         = True,
            eval_every_epoch=True,
            n_bank_pos     = n_bank_pos,
            pos_ratio      = POS_RATIO,
        )
        return best_f1

    best_cfg = manual_gru_search(_train, grid)
    print("🏅 Best GRU config:", best_cfg)

    # ---------- 5. final backbone training -----------------------
    tag = f"gru_h{best_cfg['hidden_size']}_l{best_cfg['num_layers']}"
    best_f1, backbone = train_gru(
        best_cfg,
        loaders        = (train_iter, val_once),
        input_size     = input_size,
        tag            = tag,
        resume         = False,
        eval_every_epoch=False,
        n_bank_pos     = n_bank_pos,
        pos_ratio      = POS_RATIO,
    )
    print(f"👍 Final val F1 = {best_f1:.4f}")

    export_model(backbone, "/app/models/gru_trained_model.pth")
    evaluate_and_export(backbone, [(val_X, val_y)], "gru", device)

    # ---------- 6. hybrid fine-tune ------------------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid(
        "/app/models/gru_trained_model.pth",
        loaders      = (train_iter, val_once),
        input_size   = input_size,
        hidden_size  = best_cfg["hidden_size"],
        num_layers   = best_cfg["num_layers"],
        epochs       = 3,
        lr           = 1e-3,
    )
    evaluate_and_export(hybrid, [(val_X, val_y)], "gru_hybrid", device)

# ------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
