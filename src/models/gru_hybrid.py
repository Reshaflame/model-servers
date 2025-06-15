# src/models/gru_hybrid.py
import os, logging, torch, numpy as np
import torch.nn as nn
from utils.metrics import Metrics
from utils.evaluator import quick_f1

LOG_DIR = "/workspace/logs"
CKPT_DIR = "/workspace/checkpoints"
os.makedirs(LOG_DIR, exist_ok=True);  os.makedirs(CKPT_DIR, exist_ok=True)

def _setup_logger(name="GRU", logfile="gru_training.log"):
    logger = logging.getLogger(name)
    if logger.handlers:                # already initialised
        return logger
    fmt = "%(asctime)s [%(levelname)s] " + name + " – %(message)s"
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOG_DIR, logfile)); fh.setFormatter(logging.Formatter(fmt))
    sh = logging.StreamHandler();                      sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

LOGGER = _setup_logger()

# --------------------------------------------------------------------- #
# 1.  Basic GRU (1-2 layers, 48-64 units) + checkpoint helpers          #
# --------------------------------------------------------------------- #
class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=0.2 if num_layers > 1 else 0,
                          batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):                      # x → [B, T, F]  or  [B, F]
        if x.dim() == 2: x = x.unsqueeze(1)    # add dummy time step
        h, _ = self.gru(x.contiguous())
        return self.fc(self.dropout(h[:, -1])) # logits  (sigmoid applied later)

# small helpers
def _ckpt_path(tag): return os.path.join(CKPT_DIR, f"{tag}.pt")

def save_ckpt(model, optim, epoch, tag):
    torch.save({"model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch}, _ckpt_path(tag))
    LOGGER.info(f"💾  Saved checkpoint: {_ckpt_path(tag)}")

def load_ckpt(model, optim, tag):
    path = _ckpt_path(tag)
    if not os.path.exists(path): return 0
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optim: optim.load_state_dict(state["optim"])
    LOGGER.info(f"🔁  Resumed from checkpoint: {path}")
    return state["epoch"]

# --------------------------------------------------------------------- #
# 2.  Hybrid model = frozen GRU ➊  → bottleneck 1×8 ➋ → MLP head ➌     #
# --------------------------------------------------------------------- #
class GRUHybrid(nn.Module):
    """
    ➊ Pre-trained GRU backbone (frozen)  
    ➋ 1×8 bottleneck linear layer (frozen as requested)  
    ➌ Small MLP head (trainable)
    """
    def __init__(self, gru_backbone: GRUAnomalyDetector, freeze_gru=True, freeze_bottleneck=False):
        super().__init__()
        self.gru = gru_backbone
        if freeze_gru:
            for p in self.gru.parameters():
                p.requires_grad_(False)

        self.bottleneck = nn.Linear(self.gru.fc.in_features, 8)
        if freeze_bottleneck:
            for p in self.bottleneck.parameters():
                p.requires_grad_(False)


        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        logits_gru = self.gru(x)
        feats = torch.sigmoid(logits_gru)
        feats8 = self.bottleneck(feats)
        return self.head(feats8)               

# --------------------------------------------------------------------- #
# 3.  Unified train routine                                              #
# --------------------------------------------------------------------- #
def train_gru(config, loaders, input_size, tag, resume=True, eval_every_epoch=True):
    train_loader, val_loader_fn = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUAnomalyDetector(input_size,
                               hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # imbalance handling
    n_pos = n_neg = 0
    for _, y in train_loader():                # generator – no parentheses
        n_pos += (y == 1).sum().item(); n_neg += (y == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start = load_ckpt(model, optim, tag) if resume else 0
    best_f1, patience = 0, 0
    for epoch in range(start, config["epochs"]):
        model.train()
        for xb, yb in train_loader():
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            # make sure yb is [B, 1]
            if yb.dim() == 1:          # [B]  → [B,1]
                yb = yb.unsqueeze(1)
            elif yb.dim() == 2 and yb.size(1) > 1:  # [B,1,1]  → [B,1]
                yb = yb.view(-1, 1)

            if xb.dim() == 2:          # keep the old dummy-time-step logic
                xb = xb.unsqueeze(1)

            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()

        LOGGER.info(f"[{tag}] epoch {epoch+1}/{config['epochs']} – loss {loss.item():.4f}")
        save_ckpt(model, optim, epoch+1, tag)

        if eval_every_epoch:
            mets = quick_f1(model, val_loader_fn, device)
            LOGGER.info(f"   F1={mets['F1']:.4f}  P={mets['Precision']:.4f}  R={mets['Recall']:.4f}")
            if mets['F1'] > best_f1: best_f1, patience = mets['F1'], 0
            else:
                patience += 1
                if patience >= config["early_stop_patience"]:
                    LOGGER.info("🛑  early-stopping")
                    break
    return model

def train_hybrid(backbone_ckpt, loaders, tag="gru_hybrid", epochs=3, lr=1e-3):
    train_loader, val_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ➊ load frozen backbone
    dummy_gru = GRUAnomalyDetector(1,1)        # placeholder to load weights
    dummy_gru.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
    backbone = dummy_gru.to(device)
    backbone.eval()

    model = GRUHybrid(backbone, freeze_gru=True, freeze_bottleneck=False)
    # For FastAPI retraining: (Don't forget!)
    # model = GRUHybrid(backbone, freeze_gru=True, freeze_bottleneck=True)
    optim = torch.optim.Adam(model.head.parameters(), lr=lr)    # only head trainable
    crit  = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader():
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)
            elif yb.dim() == 2 and yb.size(1) > 1:
                yb = yb.view(-1, 1)
            optim.zero_grad(); loss = crit(model(xb), yb); loss.backward(); optim.step()
        LOGGER.info(f"[HYBRID] epoch {ep+1}/{epochs} – loss {loss.item():.4f}")
    torch.save(model.state_dict(), "/app/models/gru_hybrid.pth")
    LOGGER.info("✅  hybrid model saved → /app/models/gru_hybrid.pth")
    return model
