# src/models/lstm_hybrid.py
import os, logging, torch
import torch.nn as nn
from utils.evaluator import quick_f1

# ──────────────────────────────  logging / ckpt helpers  ───────────────────── #
LOG_DIR  = "/workspace/logs"
CKPT_DIR = "/workspace/checkpoints"
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

def _ckpt(tag):            return os.path.join(CKPT_DIR, f"{tag}.pt")
def _logger(name):         # one-liner pragma
    L = logging.getLogger(name)
    if not L.handlers:
        fmt = "%(asctime)s [%(levelname)s] "+name+" – %(message)s"
        L.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log")); fh.setFormatter(logging.Formatter(fmt))
        sh = logging.StreamHandler();                                     sh.setFormatter(logging.Formatter(fmt))
        L.addHandler(fh);  L.addHandler(sh)
    return L
LOGGER = _logger("LSTM")
from utils.logging_helpers import enable_full_debug
enable_full_debug(LOGGER)        

# ───────────────────────────────  1) backbone  ─────────────────────────────── #
class LSTMRNNBackbone(nn.Module):
    """
    Original LSTM ➜ RNN stack **without sigmoid**.
    Kept identical to the code in lstm_rnn.py (just renamed).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, out=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0)
        self.rnn  = nn.RNN (hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_size, out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        rnn_out , _ = self.rnn(lstm_out)
        h_last = rnn_out[:, -1]           # (B, hidden_size)
        logits = self.fc(h_last)          # (B, 1)
        return logits, h_last             # <-- tuple

# ───────────────────────────────  2) hybrid head  ──────────────────────────── #
class LSTMHybrid(nn.Module):
    """
    ➊ Frozen LSTM+RNN backbone  
    ➋ 1×8 bottleneck (optionally frozen later on FastAPI)  
    ➌ Small MLP head that we fine-tune in stage-2
    """
    def __init__(self,
                 backbone: LSTMRNNBackbone,
                 hidden_size: int,
                 freeze_backbone: bool = True,
                 freeze_bottleneck: bool = False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad_(False)

        self.bottleneck = nn.Linear(hidden_size, 8)
        if freeze_bottleneck:
            for p in self.bottleneck.parameters(): p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.ReLU(),              # 8 → 16 → 1
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        logits, h = self.backbone(x)          # (B,1) , (B,hidden)
        feats8 = self.bottleneck(h)           # bottleneck on hidden vector
        return self.head(feats8)

# ───────────────────────────────  3) training loops  ───────────────────────── #
def _imbalance_weight(loader_fn, device):
    pos = neg = 0
    for _, y in loader_fn():
        pos += (y == 1).sum().item();  neg += (y == 0).sum().item()
    return torch.tensor([neg / max(1,pos)], device=device)

def train_lstm(cfg, loaders, input_size, tag, resume=True, eval_every=True):
    train_loader, val_loader_fn = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRNNBackbone(input_size, cfg["hidden_size"], cfg["num_layers"]).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit  = nn.BCEWithLogitsLoss(pos_weight=_imbalance_weight(train_loader, dev))

    # ── resume support ─────────────────────────────────────────────
    start = 0
    if resume and os.path.exists(_ckpt(tag)):
        state = torch.load(_ckpt(tag), map_location="cpu")
        model.load_state_dict(state["model"]); optim.load_state_dict(state["optim"])
        start = state["epoch"]; LOGGER.info(f"🔁 Resumed «{tag}» at epoch {start}")
    # ───────────────────────────────────────────────────────────────
    best_f1 = 0; patience = 0
    for ep in range(start, cfg["epochs"]):
        model.train()
        batch_id = 0
        for xb, yb in train_loader():
            batch_id += 1
            xb, yb = xb.to(dev).float(), yb.to(dev).float()
            if yb.dim()==1: yb = yb.unsqueeze(1)
            optim.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

            # ── tiny DEBUG probe every ~200 batches ─────────────────
            if batch_id % 200 == 0:
                with torch.no_grad():
                    mean_logit = torch.sigmoid(logits).mean().item()
                pos = int(yb.sum().item())
                LOGGER.debug(f"      batch {batch_id:>5}  pos={pos:>3}/{yb.numel()}  "
                             f"mean_sigmoid={mean_logit:.6f}  loss={loss.item():.6f}")
        LOGGER.info(f"[{tag}] ep {ep+1}/{cfg['epochs']} loss={loss.item():.5f}")
        torch.save({"model":model.state_dict(),"optim":optim.state_dict(),"epoch":ep+1}, _ckpt(tag))

        if eval_every:
            mets = quick_f1(model, val_loader_fn, dev)
            LOGGER.info(f"   F1={mets['F1']:.6f} P={mets['Precision']:.6f} R={mets['Recall']:.6f}")
            if mets['F1'] > best_f1: best_f1, patience = mets['F1'], 0
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"]:
                    LOGGER.info("🛑 early-stop"); break
    return best_f1, model

def train_hybrid(
        backbone_ckpt,
        loaders,
        input_size,
        hidden_size,
        num_layers,
        tag="lstm_hybrid",
        epochs=3,
        lr=1e-3):
    train_loader, val_loader = loaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ➊ load pre-trained backbone
    dummy = LSTMRNNBackbone(input_size, hidden_size, num_layers)
    dummy.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
    backbone = dummy.to(dev).eval()

    model = LSTMHybrid(
        backbone,
        hidden_size=hidden_size,
        freeze_backbone=True,
        freeze_bottleneck=False,
    ).to(dev)
    optim = torch.optim.Adam(model.head.parameters(), lr=lr)
    crit  = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader():
            xb, yb = xb.to(dev).float(), yb.to(dev).float()
            if yb.dim()==1: yb = yb.unsqueeze(1)
            optim.zero_grad(); loss=crit(model(xb), yb); loss.backward(); optim.step()
        LOGGER.info(f"[HYB] ep {ep+1}/{epochs} loss={loss.item():.5f}")
    torch.save(model.state_dict(), "/app/models/lstm_hybrid.pth")
    LOGGER.info("✅ hybrid model saved → /app/models/lstm_hybrid.pth")
    return model
