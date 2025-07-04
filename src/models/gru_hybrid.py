# ─── src/models/gru_hybrid.py ───────────────────────────────────────
import os, logging, torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from utils.evaluator import quick_f1
from utils.logging_helpers import enable_full_debug

LOG_DIR  = "/workspace/logs"
CKPT_DIR = "/workspace/checkpoints"
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# 0.  Logger helper                                                  #
# ------------------------------------------------------------------ #
def _setup_logger(name: str = "GRU", logfile: str = "gru_training.log"):
    L = logging.getLogger(name)
    if not L.handlers:                               # first initialisation
        fmt = "%(asctime)s [%(levelname)s] " + name + " – %(message)s"
        L.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(LOG_DIR, logfile))
        fh.setFormatter(logging.Formatter(fmt))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt))
        L.addHandler(fh);  L.addHandler(sh)
    return L

LOGGER = _setup_logger()
enable_full_debug(LOGGER)

# ─── imbalance helper (no DataLoader scan) ─────────────────────────
def compute_bank_pos_weight(n_pos: int, pos_ratio: float, device):
    """
    Given:
      n_pos      – number of positive sequences in the anomaly bank
      pos_ratio  – target share of positives the FastBalancedDS produces
    returns:
      1-element tensor suitable for BCEWithLogitsLoss(pos_weight=…)
    """
    n_neg = int(n_pos * (1 - pos_ratio) / pos_ratio)
    return torch.tensor([n_neg / max(1, n_pos)], device=device)


# ------------------------------------------------------------------ #
# 1.  Backbone – GRU + FC                                            #
# ------------------------------------------------------------------ #
class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 output_size: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=0.2 if num_layers > 1 else 0.)
        self.post = nn.Sequential(             # ➊ tiny two-layer “booster”
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fc  = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() == 2:            # [B, F]  →  [B, 1, F]
            x = x.unsqueeze(1)
        h, _ = self.gru(x.contiguous())
        h_last = h[:, -1]                      # (B, hidden)

        # ➋ residual + layer-norm trick
        h_boost = self.post(h_last)
        h_last  = h_last + h_boost

        logits = self.fc(self.dropout(h_last))
        return logits, h_last       # tuple, like before

# tiny helpers
def _ckpt_path(tag):                 return os.path.join(CKPT_DIR, f"{tag}.pt")
def _save_ckpt(model, optim, ep, tag):
    torch.save({"model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": ep}, _ckpt_path(tag))
    LOGGER.info(f"💾  Saved checkpoint: {_ckpt_path(tag)}")

def _load_ckpt(model, optim, tag):
    path = _ckpt_path(tag)
    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optim:
        optim.load_state_dict(state["optim"])
    LOGGER.info(f"🔁  Resumed from checkpoint: {path}")
    return state["epoch"]

def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    bce_raw = bce(logits, targets, reduction='none')
    p_t     = torch.exp(-bce_raw)
    loss    = alpha * (1-p_t) ** gamma * bce_raw
    return loss.mean()

# ------------------------------------------------------------------ #
# 2.  Hybrid head                                                    #
# ------------------------------------------------------------------ #
class GRUHybrid(nn.Module):
    """
    Frozen GRU backbone ➊ → 1×8 bottleneck ➋ → tiny MLP head ➌
    """
    def __init__(self, gru_backbone: GRUAnomalyDetector,
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
        z8 = self.bottleneck(h)
        return self.head(z8)        # final logits

# ------------------------------------------------------------------ #
# 3.  Training helpers                                               #
# ------------------------------------------------------------------ #
def _compute_pos_weight(label_array, device):
    """Fast helper for BCEWithLogitsLoss(pos_weight=…)."""
    n_pos = int(label_array.sum())
    n_neg = len(label_array) - n_pos
    return torch.tensor([n_neg / max(1, n_pos)], device=device)



def train_gru(config: dict,
              loaders,
              input_size: int,
              tag: str,
              resume: bool = True,
              eval_every_epoch: bool = True,
              *,                     
              n_bank_pos: int | None = None,
              pos_ratio:  float | None = None):
    """
    If `label_array` (np.ndarray of 0/1) is provided we compute pos_weight
    in O(1).  Otherwise we fall back to the old *slow* sweep over the
    training generator – useful for SequenceChunkedDataset.
    """
    train_loader, val_loader_fn = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUAnomalyDetector(input_size,
                               hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"]).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # ---------- imbalance handling ---------------------------------
    if n_bank_pos is not None and pos_ratio is not None:
        pos_weight = compute_bank_pos_weight(n_bank_pos, pos_ratio, dev)
    else:                                     # fallback (slow)
        n_pos = n_neg = 0
        for _, y in train_loader():
            n_pos += int((y == 1).sum())
            n_neg += int((y == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=dev)
    crit = focal_loss
    # ---------------------------------------------------------------

    start_ep = _load_ckpt(model, optim, tag) if resume else 0
    best_f1, patience = 0., 0
    for ep in range(start_ep, config["epochs"]):
        model.train()
        for batch_id, (xb, yb) in enumerate(train_loader()):
            xb = xb.to(dev).float()
            yb = yb.to(dev).float()
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)

            optim.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

            if batch_id % 200 == 0:
                with torch.no_grad():
                    mean_sig = torch.sigmoid(logits).mean().item()
                LOGGER.debug(f"batch {batch_id:>5}  "
                             f"mean_sigmoid={mean_sig:.6f}  "
                             f"loss={loss.item():.6f}")

        LOGGER.info(f"[{tag}] epoch {ep+1}/{config['epochs']} – loss {loss.item():.6f}")
        _save_ckpt(model, optim, ep + 1, tag)

        if eval_every_epoch:
            mets = quick_f1(model, val_loader_fn, dev)
            LOGGER.info(f"   F1={mets['F1']:.6f}  "
                        f"P={mets['Precision']:.6f}  R={mets['Recall']:.6f}")
            if mets["F1"] > best_f1:
                best_f1, patience = mets["F1"], 0
            else:
                patience += 1
                if patience >= config["early_stop_patience"]:
                    LOGGER.info("🛑 early-stopping")
                    break
    return best_f1, model


def train_hybrid(backbone_ckpt: str,
                 loaders,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 tag: str = "gru_hybrid",
                 epochs: int = 3,
                 lr: float = 1e-3):
    """Stage-2 fine-tune just the MLP head."""
    train_loader, _ = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ➊ load frozen backbone
    dummy = GRUAnomalyDetector(input_size, hidden_size, num_layers)
    dummy.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
    backbone = dummy.to(dev).eval()

    model = GRUHybrid(backbone, hidden_size,
                      freeze_gru=True, freeze_bottleneck=False).to(dev)
    optim = torch.optim.Adam(model.head.parameters(), lr=lr)
    crit = focal_loss

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader():
            xb = xb.to(dev).float()
            yb = yb.to(dev).float()
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)

            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
        LOGGER.info(f"[HYBRID] epoch {ep+1}/{epochs} – loss {loss.item():.6f}")

    torch.save(model.state_dict(), "/app/models/gru_hybrid.pth")
    LOGGER.info("✅ hybrid model saved → /app/models/gru_hybrid.pth")
    return model
