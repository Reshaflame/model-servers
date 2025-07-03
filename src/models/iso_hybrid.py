# src/models/iso_hybrid.py
import torch
import torch.nn as nn
from .iso_backbone import IsoBackbone

class IsoHybrid(nn.Module):
    """
    ➊ Frozen Isolation-Forest backbone (scikit).
    ➋ 1×8 linear bottleneck (trainable or frozen).
    ➌ MLP head → final anomaly logit.
    """
    def __init__(self,
                 iso_backbone: IsoBackbone,
                 freeze_bottleneck=False):
        super().__init__()

        self.backbone = iso_backbone           # sklearn object (frozen)

        # ➋ interface: 1-D score  →  8-D feature vector
        self.bottleneck = nn.Linear(1, 8)
         # ➌ tiny “booster” (+ residual) – mirrors GRU/LSTM
        self.post = nn.Sequential(
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )
        self.dropout = nn.Dropout(0.2)

        # ➍ small MLP head
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)          # logit
        )

    # --------------------------------------------------
    # x must be a N×F *numpy* array (not tensor) because
    # the sklearn backbone expects numpy.
    # --------------------------------------------------
    def forward(self, x_numpy):
        with torch.no_grad():
            score = self.backbone.score(x_numpy)         # shape (N,)
        device = self.bottleneck.weight.device
        score_tensor = torch.as_tensor(score, dtype=torch.float32, device=device)\
                           .unsqueeze(1)                 # (N,) → (N,1)
        feats8 = self.bottleneck(score_tensor)           # (N,8)
        feats8 = feats8 + self.post(feats8)              # ⚡ residual boost
        return self.head(self.dropout(feats8))           # logits

# ---------------------------------------------------------------------- #
# Loss helper (shared with GRU/LSTM)
# ---------------------------------------------------------------------- #
def focal_loss(logits, targets, alpha: float = 1.0, gamma: float = 2.0):
    from torch.nn.functional import binary_cross_entropy_with_logits as _bce
    targets = targets.to(logits.device)
    raw = _bce(logits, targets, reduction="none")
    p_t = torch.exp(-raw)
    return (alpha * (1 - p_t) ** gamma * raw).mean()