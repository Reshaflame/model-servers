# ---------------------------------------------------------------
# Isolation-Forest ➜ bottleneck ➜ MLP hybrid fine-tune pipeline
# ---------------------------------------------------------------
# ➊ Train / resume an Isolation-Forest backbone
# ➋ Freeze backbone, fine-tune a tiny head with focal-loss
# ➌ Evaluate on the full labelled stream (row-order irrelevant)
# ---------------------------------------------------------------

import json, logging, os, torch, numpy as np
from pathlib import Path

from utils.constants          import CHUNKS_LABELED_PATH
from utils.fast_balanced_dataset import FastBalancedDS
from utils.build_anomaly_bank import build_if_needed as build_pos_bank
from utils.build_negative_bank import build_if_needed as build_neg_bank

from models.iso_backbone      import IsoBackbone
from models.iso_hybrid        import IsoHybrid, focal_loss

# ─────────── log / checkpoint dirs (env-override) ──────────────
LOG_DIR  = os.getenv("LOG_DIR",  "/workspace/logs")
CKPT_DIR = os.getenv("CKPT_DIR", "/workspace/checkpoints")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[IsoHybrid] %(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "iso_hybrid.log")),
              logging.StreamHandler()]
)

# ───────────── fixed constants (seq_len = 1) ────────────────────
BANK_DIR   = "/workspace/model-servers/data/iso_seq1"
os.makedirs(BANK_DIR, exist_ok=True)
BANK_PT_POS = f"{BANK_DIR}/anomaly_bank.pt"
BANK_PT_NEG = f"{BANK_DIR}/negative_bank.pt"

POS_RATIO   = 0.30
BATCH       = 64
N_WORKERS   = 4
ISO_PT      = Path("/app/models/iso_backbone.joblib")
HYB_PT      = Path("/app/models/iso_hybrid.pth")

# ===============================================================
def run_iso_hybrid_pipeline() -> None:
    # 0️⃣  canonical feature list ------------------------------------------------
    with open("data/meta/expected_features.json") as f:
        FEATURES = json.load(f)
    logging.info(f"🔍 Using {len(FEATURES)} features from expected_features.json")

    # 1️⃣  build / refresh anomaly & negative banks -----------------------------
    build_pos_bank(CHUNKS_LABELED_PATH, BANK_PT_POS,
                   feature_cols=FEATURES, seq_len=1)
    build_neg_bank(CHUNKS_LABELED_PATH, BANK_PT_NEG,
                   feature_cols=FEATURES, seq_len=1)

    # fast, balanced loader – rows are i.i.d. (sequence length = 1)
    ds_full = FastBalancedDS(
        chunk_dir    = CHUNKS_LABELED_PATH,
        bank_pt      = BANK_PT_POS,
        neg_bank_pt  = BANK_PT_NEG,
        feature_cols = FEATURES,
        seq_len      = 1,
        pos_ratio    = POS_RATIO,
    )

    train_loader = torch.utils.data.DataLoader(
        ds_full, batch_size=BATCH, shuffle=True,
        num_workers=N_WORKERS, pin_memory=True
    )

    eval_loader = torch.utils.data.DataLoader(
        ds_full, batch_size=BATCH, shuffle=False,
        num_workers=N_WORKERS, pin_memory=True
    )

    # 2️⃣  Isolation-Forest backbone (fit once, smart-resume) -------------------
    if ISO_PT.exists():
        logging.info("ℹ️  iso_backbone.joblib found – loading.")
        iso_backbone = IsoBackbone.load(str(ISO_PT))
    else:
        logging.info("🚀  Fitting Isolation-Forest backbone …")
        # sample up to ~1 M rows (200 * 5 000) for speed
        X_train, max_chunks = [], 200
        for i, (xb, _) in enumerate(train_loader):
            X_train.append(xb.squeeze(1).cpu().numpy())
            if i + 1 >= max_chunks:
                break
        X_train = np.concatenate(X_train, axis=0)
        logging.info(f"🧠  Backbone fit on {len(X_train):,} samples")
        iso_backbone = IsoBackbone(contamination=0.05, n_estimators=100)
        iso_backbone.fit(X_train)
        iso_backbone.save(str(ISO_PT))
        logging.info("✅  Saved iso_backbone.joblib")

    # 3️⃣  Hybrid fine-tune ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IsoHybrid(iso_backbone, freeze_bottleneck=False).to(device)

    if HYB_PT.exists():
        logging.info("ℹ️  iso_hybrid.pth found – reloading weights.")
        model.load_state_dict(torch.load(str(HYB_PT), map_location=device))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 3
    for ep in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            feats_np = xb.squeeze(1).cpu().numpy()           # (N,F)
            yb = yb.float().view(-1, 1).to(device)           # (N,1)

            optim.zero_grad()
            logits = model(feats_np)
            loss   = focal_loss(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item()

        logging.info(f"[Epoch {ep+1}/{EPOCHS}] loss = {loss_sum:.6f}")

    torch.save(model.state_dict(), str(HYB_PT))
    logging.info("✅  Saved iso_hybrid.pth")

    # 4️⃣  Evaluation (row-wise) -----------------------------------------------
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for xb, yb in eval_loader:
            feats_np = xb.squeeze(1).cpu().numpy()
            logits   = model(feats_np)
            preds    = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().ravel()
            labels   = yb.int().cpu().numpy().ravel()

            tp += np.logical_and(preds == 1, labels == 1).sum()
            fp += np.logical_and(preds == 1, labels == 0).sum()
            fn += np.logical_and(preds == 0, labels == 1).sum()

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    logging.info(f"📊  Hybrid metrics → P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")
    logging.info("🎯  IsoHybrid pipeline finished.")


# -------------------------------------------------------------------
if __name__ == "__main__":
    run_iso_hybrid_pipeline()
