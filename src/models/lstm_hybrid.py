# ─── src/models/lstm_hybrid.py ──────────────────────────────────────
import os, logging, torch
import torch.nn as nn
from utils.evaluator  import quick_f1
from utils.logging_helpers import enable_full_debug

LOG_DIR, CKPT_DIR = "/workspace/logs", "/workspace/checkpoints"
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# 0.  Logging helpers                                                #
# ------------------------------------------------------------------ #
def _ckpt(tag):  return os.path.join(CKPT_DIR, f"{tag}.pt")

def _logger(name="LSTM"):
    L = logging.getLogger(name)
    if not L.handlers:
        fmt = "%(asctime)s [%(levelname)s] " + name + " – %(message)s"
        L.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
        fh.setFormatter(logging.Formatter(fmt))
        sh = logging.StreamHandler(); sh.setFormatter(logging.Formatter(fmt))
        L.addHandler(fh); L.addHandler(sh)
    return L

LOGGER = _logger()
enable_full_debug(LOGGER)

# ------------------------------------------------------------------ #
# 1.  Backbone: LSTM ➜ RNN ➜ FC                                      #
# ------------------------------------------------------------------ #
class LSTMRNNBackbone(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 out: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.)
        self.rnn  = nn.RNN (hidden_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.)
        self.fc   = nn.Linear(hidden_size, out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)              # [B, F] → [B, 1, F]
        lstm_out, _ = self.lstm(x)
        rnn_out,  _ = self.rnn(lstm_out)
        h_last = rnn_out[:, -1]             # (B, hidden)
        logits = self.fc(h_last)            # (B, 1)
        return logits, h_last               # tuple (unchanged)

# ------------------------------------------------------------------ #
# 2.  Hybrid head                                                    #
# ------------------------------------------------------------------ #
class LSTMHybrid(nn.Module):
    def __init__(self, backbone: LSTMRNNBackbone,
                 hidden_size: int,
                 freeze_backbone: bool = True,
                 freeze_bottleneck: bool = False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
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
        logits, h = self.backbone(x)
        z8 = self.bottleneck(h)
        return self.head(z8)

# ------------------------------------------------------------------ #
# 3.  Training helpers                                               #
# ------------------------------------------------------------------ #
def _pos_weight_from_labels(label_array, device):
    n_pos = int(label_array.sum())
    n_neg = len(label_array) - n_pos
    return torch.tensor([n_neg / max(1, n_pos)], device=device)

# ------------------------------------------------------------------ #
# 4.  Stage-1 backbone training                                      #
# ------------------------------------------------------------------ #
def train_lstm(cfg: dict,
               loaders,
               input_size: int,
               tag: str,
               resume: bool = True,
               eval_every: bool = True,
               label_array=None):
    train_loader, val_loader_fn = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRNNBackbone(input_size,
                            hidden_size=cfg["hidden_size"],
                            num_layers=cfg["num_layers"]).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # pos_weight – fast path if we have the global labels array
    if label_array is not None:
        pos_weight = _pos_weight_from_labels(label_array, dev)
    else:                                 # fallback (may be slow)
        pos = neg = 0
        for _, y in train_loader():
            pos += (y == 1).sum().item();  neg += (y == 0).sum().item()
        pos_weight = torch.tensor([neg / max(1, pos)], device=dev)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start = 0
    if resume and os.path.exists(_ckpt(tag)):
        state = torch.load(_ckpt(tag), map_location="cpu")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start = state["epoch"]
        LOGGER.info(f"🔁 Resumed «{tag}» at epoch {start}")

    best_f1, patience = 0., 0
    for ep in range(start, cfg["epochs"]):
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

        LOGGER.info(f"[{tag}] ep {ep+1}/{cfg['epochs']} loss={loss.item():.5f}")
        torch.save({"model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": ep + 1}, _ckpt(tag))

        if eval_every:
            mets = quick_f1(model, val_loader_fn, dev)
            LOGGER.info(f"   F1={mets['F1']:.6f}  "
                        f"P={mets['Precision']:.6f}  R={mets['Recall']:.6f}")
            if mets["F1"] > best_f1:
                best_f1, patience = mets["F1"], 0
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"]:
                    LOGGER.info("🛑 early-stop"); break

    return best_f1, model

# ------------------------------------------------------------------ #
# 5.  Stage-2 Hybrid fine-tune                                       #
# ------------------------------------------------------------------ #
def train_hybrid(backbone_ckpt: str,
                 loaders,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 tag: str = "lstm_hybrid",
                 epochs: int = 3,
                 lr: float = 1e-3):
    train_loader, _ = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy = LSTMRNNBackbone(input_size, hidden_size, num_layers)
    dummy.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
    backbone = dummy.to(dev).eval()

    model = LSTMHybrid(backbone, hidden_size,
                       freeze_backbone=True,
                       freeze_bottleneck=False).to(dev)
    optim = torch.optim.Adam(model.head.parameters(), lr=lr)
    crit  = nn.BCEWithLogitsLoss()

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
        LOGGER.info(f"[HYB] ep {ep+1}/{epochs} loss={loss.item():.5f}")

    torch.save(model.state_dict(), "/app/models/lstm_hybrid.pth")
    LOGGER.info("✅ hybrid model saved → /app/models/lstm_hybrid.pth")
    return model
