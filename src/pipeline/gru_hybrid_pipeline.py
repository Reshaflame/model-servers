# ---------------------------------------------------------------
#  GRU + MLP hybrid training pipeline  (FAST balanced dataset)
# ---------------------------------------------------------------
from glob import glob
import os, random, torch, pandas as pd
import numpy as np, logging, json
from torch.utils.data import DataLoader
from pathlib import Path

from utils.fast_balanced_dataset import FastBalancedDS                # ⬅ RAM-banks dataset
from utils.build_anomaly_bank    import build_if_needed as build_pos_bank
from utils.build_negative_bank   import build_if_needed as build_negative_bank
from utils.dl_helpers            import stack_batch, to_device
from utils.constants             import CHUNKS_LABELED_PATH
from utils.tuning                import manual_gru_search
from utils.evaluator             import quick_f1, evaluate_and_export
from utils.model_exporter        import export_model
from models.gru_hybrid           import train_gru, train_hybrid, GRUAnomalyDetector

# ------------------------------------------------------------------ #
BANK_PT_POS = "/workspace/model-servers/data/anomaly_bank.pt"
BANK_PT_NEG = "/workspace/model-servers/data/negative_bank.pt"
POS_RATIO   = 0.30                 # ≈ 30 % positives per mini-batch
BATCH_SIZE  = 64
NUM_WORKERS = 4                    # ← adjust to your CPU
FINAL_MODEL_PT = Path("/app/models/gru_trained_model.pth")  # backbone
FINAL_HYB_PT   = Path("/app/models/gru_hybrid.pth")         # hybrid (optional)

# ------------------------------------------------------------------ #
def run_pipeline() -> None:
    # ---------- 0. metadata ---------------------------------------
    chunk_dir  = CHUNKS_LABELED_PATH

    # ---- SINGLE source-of-truth for the schema -----------------
    with open("data/meta/expected_features.json") as f:
        FEATURE_LIST = json.load(f)          # 61 columns – Phase-2
    input_size = len(FEATURE_LIST)           # = 61
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1. build banks (if missing) -----------------------
    build_pos_bank(chunk_dir, BANK_PT_POS,
                   feature_cols=FEATURE_LIST, seq_len=1)
    if not os.path.exists(BANK_PT_NEG):
        build_negative_bank(
            src_dir      = CHUNKS_LABELED_PATH,
            out_pt       = BANK_PT_NEG,
            feature_cols = FEATURE_LIST,
            seq_len      = 1
        )                     # one-off – very fast

    # ---------- 2. dataset & loaders -----------------------------
    full_ds = FastBalancedDS(
        chunk_dir   = chunk_dir,
        bank_pt     = BANK_PT_POS,
        neg_bank_pt = BANK_PT_NEG,
        feature_cols= FEATURE_LIST,
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

    # ────────────────────────────────────────────────────────────
    #  A)   FIND BEST EXISTING CHECKPOINT (if any)
    # ────────────────────────────────────────────────────────────
    ckpts = sorted(glob("/workspace/checkpoints/gru_h*.pt"))
    best_ck, best_f1 = None, -1.
    for ck in ckpts:
        h = int( ck.split("_h")[1].split("_")[0] )
        l = int( ck.split("_l")[1].split(".")[0] )
        dummy = GRUAnomalyDetector(input_size, hidden_size=h, num_layers=l)
        dummy.load_state_dict(torch.load(ck, map_location="cpu")["model"])
        f1 = quick_f1(dummy, val_once, device="cpu")["F1"]
        print(f"▶ {os.path.basename(ck):<18}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_ck, best_h, best_l = f1, ck, h, l

    if best_ck is not None:
        print(f"🏅 BEST existing ckpt → {os.path.basename(best_ck)}  (F1={best_f1:.4f})")
        # fine-tune for a few *extra* epochs if you like
        EXTRA_EPOCHS = 3
        backbone = GRUAnomalyDetector(input_size, best_h, best_l)
        state = torch.load(best_ck, map_location="cpu")
        backbone.load_state_dict(state["model"])
        if EXTRA_EPOCHS:
            _, backbone = train_gru(
                {"lr":1e-3, "hidden_size":best_h, "num_layers":best_l,
                    "epochs":EXTRA_EPOCHS, "early_stop_patience":2},
                loaders        = (train_iter, val_once),
                input_size     = input_size,
                tag            = "best_ckpt_finetune",
                resume         = False,           # fresh optim
                eval_every_epoch=False,
                n_bank_pos     = n_bank_pos,
                pos_ratio      = POS_RATIO,
            )
        export_model(backbone, "/app/models/gru_trained_model.pth")
        print("✅  Exported best backbone – pipeline finished.")
        return                    # ← grid-search *skipped*

    # ──────────────────────────────────────────────────────────────
    #  ## smart-resume guard – insert HERE #########################
    # ──────────────────────────────────────────────────────────────
    if FINAL_MODEL_PT.exists():
        print("ℹ️  Found existing /app/models/gru_trained_model.pth → skipping retraining")

        # dummy backbone must match the saved checkpoint config
        backbone = GRUAnomalyDetector(input_size, hidden_size=64, num_layers=2)
        backbone.load_state_dict(torch.load(FINAL_MODEL_PT, map_location="cpu"))
        backbone.eval()

        # fresh metrics on *current* val-tensor
        mets = quick_f1(backbone, val_once, device="cpu")
        print(f"✅  F1={mets['F1']:.4f} ‖ P={mets['Precision']:.3f} R={mets['Recall']:.3f}")

        # regenerate preds so they match the new val set
        evaluate_and_export(
            backbone, [(val_X, val_y)],
            model_name="gru_recheck",
            device="cpu",
            export_ground_truth=True,
            thr=0.25,                      # tuned threshold
        )

        # optional: also skip hybrid fine-tune if it already exists
        if FINAL_HYB_PT.exists():
            print("ℹ️  gru_hybrid.pth already present – pipeline finished.")
            return

        # otherwise run only the hybrid stage with the existing backbone
        print("🚀  Existing backbone → running HYBRID stage only …")
        best_cfg = {"hidden_size": 64, "num_layers": 2} 
        hybrid = train_hybrid(
            str(FINAL_MODEL_PT),
            loaders      = (lambda: (), val_once),  # no further training of backbone
            input_size   = input_size,
            hidden_size  = best_cfg["hidden_size"],
            num_layers   = best_cfg["num_layers"],
            epochs       = 3,
            lr           = 1e-3,
        )
        evaluate_and_export(
            hybrid, [(val_X, val_y)],
            model_name="gru_hybrid_recheck",
            device="cpu",
            export_ground_truth=True,
            thr=0.25,
        )
        return          # ← finished, no grid-search / training
    # ──────────────────────────────────────────────────────────────
    #  ## end smart-resume guard ###################################
    # ──────────────────────────────────────────────────────────────


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

    # ---------- 5. pick BEST checkpoint & (optionally) resume ----
    print("\n🔎  Scanning checkpoints for best F1 …")

    def _arch_from_fname(fn: str) -> tuple[int, int]:
        # e.g.  gru_h64_l2.pt  ->  (64, 2)
        base = os.path.basename(fn)
        h = int(base.split("_h")[1].split("_")[0])
        l = int(base.split("_l")[1].split(".")[0])
        return h, l

    ckpts = sorted(glob("/workspace/checkpoints/gru_h*.pt"))
    if not ckpts:
        raise RuntimeError("No checkpoints found – did grid-search run?")

    best_f1, best_path, best_h, best_l = -1., None, None, None
    for ck in ckpts:
        h, l = _arch_from_fname(ck)
        model = GRUAnomalyDetector(input_size, hidden_size=h, num_layers=l)
        sd    = torch.load(ck, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)
        f1    = quick_f1(model, val_once, device="cpu")["F1"]
        print(f"{os.path.basename(ck):<20}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_path, best_h, best_l = f1, ck, h, l

    print(f"\n🏅  BEST checkpoint → {os.path.basename(best_path)}  (F1={best_f1:.4f})")

    # ------- optionally resume it for a few extra epochs ----------
    EXTRA_EPOCHS = 3            # set to 0 if you don’t want any resume

    backbone = GRUAnomalyDetector(input_size, best_h, best_l)
    state    = torch.load(best_path, map_location="cpu")
    backbone.load_state_dict(state["model"])
    start_ep = state["epoch"]

    if EXTRA_EPOCHS > 0:
        cfg_resume = {
            "lr": 5e-4,
            "hidden_size": best_h,
            "num_layers":  best_l,
            "epochs":      start_ep + EXTRA_EPOCHS,
            "early_stop_patience": EXTRA_EPOCHS + 1,   # disable stop
        }
        print(f"🔄  Resuming for {EXTRA_EPOCHS} extra epoch(s)…")
        _, backbone = train_gru(
            cfg_resume,
            loaders        = (train_iter, val_once),
            input_size     = input_size,
            tag            = f"best_gru_h{best_h}_l{best_l}",
            resume         = True,          # continues from checkpoint
            eval_every_epoch=False,
            n_bank_pos     = n_bank_pos,
            pos_ratio      = POS_RATIO,
        )

    # ----- export & evaluate the final backbone just like before ----
    export_model(backbone, "/app/models/gru_trained_model.pth")
    evaluate_and_export(
        backbone, [(val_X, val_y)],
        model_name="gru_final",
        device=device,
        export_ground_truth=True,
        thr=0.25,
    )


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
    evaluate_and_export(
        hybrid, [(val_X, val_y)],
        model_name="gru_hybrid",
        device=device,
        export_ground_truth=True,
        thr=0.25,
    )
    LOGGER = logging.getLogger("PIPELINE")

    pred_dir  = "/app/models/preds/gru_hybrid_batches"
    cands = sorted(glob(os.path.join(pred_dir, "*")))
    first = next(f for f in cands if f.endswith((".pt", ".npy")))

    try:                                     # works for both .npy and .pt
        if first.endswith(".npy"):
            batch = np.load(first, allow_pickle=True).item()
            y_true  = batch["y"].astype(int).squeeze()
            y_prob  = batch["y_hat"].squeeze()
        else:                                # .pt
            batch = torch.load(first, map_location="cpu")
            y_true = batch["y"].int().view(-1).numpy()
            y_prob = torch.sigmoid(batch["logits"]).view(-1).numpy()

        pos = int(y_true.sum())
        neg = len(y_true) - pos
        auc = float( (np.argsort(y_prob)[-pos:] == np.where(y_true==1)[0]).mean() )

        LOGGER.info(f"[peek] first-batch size={len(y_true)}  "
                    f"pos={pos}  neg={neg}  ‖  rough-AUC≈{auc:0.3f}")

        # optional: log first 10 triplets for sanity-check
        LOGGER.debug("[peek] y / ŷ  : " +
                    json.dumps([{"y":int(y), "p":float(p)} 
                                for y,p in zip(y_true[:10], y_prob[:10])]))

    except Exception as e:
        LOGGER.warning(f"[peek] couldn’t read {first}: {e}")
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
