# ---------------------------------------------------------------
#  LSTM+RNN + MLP hybrid training pipeline (streaming dataset)
# ---------------------------------------------------------------
import os, json, torch, re, pandas as pd
from glob import glob as _glob
from pathlib import Path
from utils.fast_balanced_dataset import FastBalancedDS
from utils.build_anomaly_bank    import build_if_needed as build_pos_bank
from utils.build_negative_bank   import build_if_needed as build_neg_bank
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning      import manual_gru_search          # reuse helper
from utils.evaluator   import evaluate_and_export, quick_f1
from utils.model_exporter import export_model
from models.lstm_hybrid   import train_lstm, train_hybrid, LSTMRNNBackbone, LSTMHybrid

# -------- constants ---------------------------------------------
BANK_DIR     = "/workspace/model-servers/data/lstm_seq10"
os.makedirs(BANK_DIR, exist_ok=True)

BANK_PT_POS  = f"{BANK_DIR}/anomaly_bank.pt"
BANK_PT_NEG  = f"{BANK_DIR}/negative_bank.pt"
POS_RATIO   = 0.30          # 30 % positives per mini-batch
BATCH_SIZE  = 64
NUM_WORKERS = 4
SEQ_LEN     = 10            # keep the 10-timestep window you used before

BACKBONE_PT = Path("/app/models/lstm_rnn_trained_model.pth")
HYBRID_PT   = Path("/app/models/lstm_hybrid.pth")

# ---------- helper -------------------------------------------------
def load_backbone_from_ckpt(pt: Path, input_size: int) -> tuple[LSTMRNNBackbone, int, int]:
    """
    Inspect the .pth → return (model, hidden_size, num_layers)
    """
    sd = torch.load(pt, map_location="cpu")

    # LSTM keys look like   'lstm.weight_ih_l0', 'lstm.weight_ih_l1', …
    lstm_keys = [k for k in sd if k.startswith("lstm.weight_ih_l")]
    num_layers = len(lstm_keys)

    # weight_ih_l0 shape is (4*hidden, input_size)
    hidden_size = sd["lstm.weight_ih_l0"].shape[0] // 4

    mdl = LSTMRNNBackbone(input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
    mdl.load_state_dict(sd)          # strict=True – all keys match
    mdl.eval()
    return mdl, hidden_size, num_layers

# --- helper: load a model .pth and print F1/P/R on the current val set
def _print_metrics(pt_file: Path, build_model_fn, val_once, name: str):
    mdl = build_model_fn().to("cpu")
    mdl.load_state_dict(torch.load(pt_file, map_location="cpu"))
    mdl.eval()
    mets = quick_f1(mdl, val_once, device="cpu")
    print(f"✅  {name:9s} F1={mets['F1']:.4f}  "
          f"P={mets['Precision']:.3f}  R={mets['Recall']:.3f}")
    return mdl            # you may want it later


def run_pipeline() -> None:
    # ---------- 0. schema & meta ---------------------------------
    with open("data/meta/expected_features.json") as f:
        FEATURE_LIST = json.load(f)          # 61 columns
    input_size = len(FEATURE_LIST)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    chunk_dir = CHUNKS_LABELED_PATH

    # ---------- 1. build / load banks ----------------------------
    build_pos_bank(chunk_dir, BANK_PT_POS,
                   feature_cols=FEATURE_LIST, seq_len=SEQ_LEN)
    build_neg_bank(chunk_dir, BANK_PT_NEG,
                   feature_cols=FEATURE_LIST, seq_len=SEQ_LEN)

    # ---------- 2. dataset (balanced) ----------------------------
    full_ds = FastBalancedDS(
        chunk_dir   = chunk_dir,
        bank_pt     = BANK_PT_POS,
        neg_bank_pt = BANK_PT_NEG,
        feature_cols= FEATURE_LIST,
        seq_len     = SEQ_LEN,
        pos_ratio   = POS_RATIO,
    )

    train_loader = lambda: torch.utils.data.DataLoader(
        full_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # one-shot validation tensor (same trick the GRU pipeline uses)
    val_X, val_y = next(iter(torch.utils.data.DataLoader(
        full_ds, batch_size=4000, shuffle=True)))
    val_once = lambda: [(val_X, val_y)]

    # full-set iterator for predictions
    full_iter = lambda: torch.utils.data.DataLoader(
        full_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)

    pos_tr  = sum((y == 1).sum().item() for _, y in train_loader())
    pos_val = sum((y == 1).sum().item() for _, y in val_once())
    print(f"🧮 train positives ≈ {pos_tr}")
    print(f"🧮   val positives ≈ {pos_val}")

    # ---------- 2. smart-resume guard --------------------------------
    if BACKBONE_PT.exists():
        print("ℹ️  Found backbone – skipping retrain")

        backbone, h, l = load_backbone_from_ckpt(BACKBONE_PT, input_size)
        mets = quick_f1(backbone, val_once, device="cpu")
        print(f"✅  Backbone  F1={mets['F1']:.4f}  "
            f"P={mets['Precision']:.3f}  R={mets['Recall']:.3f}")

        # ---------- check hybrid -------------------------------------
        if HYBRID_PT.exists():
            def _build_hybrid_skeleton() -> torch.nn.Module:
                """
                Construct **exactly** the same hybrid architecture that
                train_hybrid() produces, but without running any training.
                """
                # backbone is already built & frozen
                mdl = LSTMHybrid(backbone,
                                hidden_size=h,
                                num_layers=l,
                                freeze_lstm=True,
                                freeze_bottleneck=False)
                return mdl

            _print_metrics(HYBRID_PT, _build_hybrid_skeleton,
                        val_once, name="Hybrid")
            print("ℹ️  Both models already present – pipeline finished.")
            return


        # ---------- run hybrid only ----------------------------------
        print("🚀  Backbone ready – running hybrid stage only…")
        hybrid = train_hybrid(
            str(BACKBONE_PT),
            loaders      = (lambda: (), val_once),   # no further backbone training
            input_size   = input_size,
            hidden_size  = h,
            num_layers   = l,
            epochs       = 3,
            lr           = 1e-3,
        )
        evaluate_and_export(
            hybrid, full_iter(), model_name="lstm_hybrid", device="cpu"
        )
        _print_metrics(HYBRID_PT, _build_hybrid_skeleton,
                    val_once, name="Hybrid")
        return
    # -----------------------------------------------------------------



    # ---------- 3. grid search -----------------------------------
    grid = [
        {"lr":1e-3 , "hidden_size":64 , "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4 , "hidden_size":128, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3 , "hidden_size":64 , "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def _train(cfg):
        best_f1, _ = train_lstm(
            cfg,
            loaders=(train_loader, val_once),
            input_size=input_size,
            tag=f"lstm_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every_epoch=True,
        )
        return best_f1

    best = manual_gru_search(_train, grid)
    print("🏅 Best LSTM config:", best)

    # ---------- 4. final backbone train --------------------------
    tag = f"lstm_h{best['hidden_size']}_l{best['num_layers']}"
    best_f1, lstm_backbone = train_lstm(
        best,
        loaders=(train_loader, val_once),
        input_size=input_size,
        tag=tag,
        resume=False,
        eval_every_epoch=False,
    )
    print(f"👍 Final backbone F1 = {best_f1:.4f}")

    export_model(lstm_backbone, str(BACKBONE_PT))
    evaluate_and_export(
        lstm_backbone, full_iter(), model_name="lstm",
        device=device, export_ground_truth=True
    )

    # ---------- 5. hybrid fine-tune ------------------------------
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid = train_hybrid(
        str(BACKBONE_PT),
        loaders=(train_loader, val_once),
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
