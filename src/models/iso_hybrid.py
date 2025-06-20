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
        if freeze_bottleneck:
            for p in self.bottleneck.parameters():
                p.requires_grad_(False)

        # ➌ small MLP head
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 1)          # logit
        )

    # --------------------------------------------------
    # x must be a N×F *numpy* array (not tensor) because
    # the sklearn backbone expects numpy.
    # --------------------------------------------------
    def forward(self, x_numpy):
        with torch.no_grad():
            score = self.backbone.score(x_numpy)         # shape (N,)
        score_tensor = torch.tensor(score, dtype=torch.float32,
                                    device=self.head[0].weight.device).unsqueeze(1)
        feats8 = self.bottleneck(score_tensor)
        return self.head(feats8)                         # logits
