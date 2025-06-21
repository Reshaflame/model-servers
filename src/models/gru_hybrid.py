# src/models/gru_hybrid.py
import os, logging, torch
import torch.nn as nn
from utils.evaluator import quick_f1
from utils.logging_helpers import enable_full_debug

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
enable_full_debug(LOGGER)

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

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h, _ = self.gru(x.contiguous())
        h_last = h[:, -1]                 # (B, hidden_size)
        logits = self.fc(self.dropout(h_last))
        return logits, h_last

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
    def __init__(self,
                 gru_backbone: GRUAnomalyDetector,
                 hidden_size: int,
                 freeze_gru: bool = True,
                 freeze_bottleneck: bool = False):
        super().__init__()
        self.gru = gru_backbone
        if freeze_gru:
            for p in self.gru.parameters():
                p.requires_grad_(False)

        self.bottleneck = nn.Linear(hidden_size, 8)
        if freeze_bottleneck:
            for p in self.bottleneck.parameters():
                p.requires_grad_(False)


        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        logits, h = self.gru(x)
        feats8 = self.bottleneck(h)
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
        batch_id = 0
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
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            # -- light debug every ~200 mini-batches -----------------
            if batch_id % 200 == 0:
                with torch.no_grad():
                    mean_logit = torch.sigmoid(logits).mean().item()
                pos = int(yb.sum().item())
                LOGGER.debug(f"      batch {batch_id:>4}  pos={pos:>3}/{yb.numel()} "
                             f"mean_sigmoid={mean_logit:.6f}")
            batch_id += 1

        LOGGER.info(f"[{tag}] epoch {epoch+1}/{config['epochs']} – loss {loss.item():.6f}")
        save_ckpt(model, optim, epoch+1, tag)

        if eval_every_epoch:
            mets = quick_f1(model, val_loader_fn, device)
            LOGGER.info(f"   F1={mets['F1']:.6f}  P={mets['Precision']:.6f}  R={mets['Recall']:.6f}")
            if mets['F1'] > best_f1: best_f1, patience = mets['F1'], 0
            else:
                patience += 1
                if patience >= config["early_stop_patience"]:
                    LOGGER.info("🛑  early-stopping")
                    break
    return best_f1, model

def train_hybrid(
        backbone_ckpt,
        loaders,
        input_size,
        hidden_size,
        num_layers,
        tag="gru_hybrid",
        epochs=3,
        lr=1e-3):
    train_loader, val_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ➊ load frozen backbone with real dims
    dummy_gru = GRUAnomalyDetector(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    dummy_gru.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
    backbone = dummy_gru.to(device)
    backbone.eval()

    model = GRUHybrid(
        backbone,
        hidden_size=hidden_size,
        freeze_gru=True,
        freeze_bottleneck=False,
    ).to(device)
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
        LOGGER.info(f"[HYBRID] epoch {ep+1}/{epochs} – loss {loss.item():.6f}")
    torch.save(model.state_dict(), "/app/models/gru_hybrid.pth")
    LOGGER.info("✅  hybrid model saved → /app/models/gru_hybrid.pth")
    return model
