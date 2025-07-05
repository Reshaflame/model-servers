# ------------------------------------------------------------------
#  Helper: score every *.pt checkpoint and return best path & score
# ------------------------------------------------------------------
import os, torch
from glob import glob
from typing import Tuple
from models.gru_hybrid import GRUAnomalyDetector
from utils.evaluator import quick_f1

def best_ckpt(path_pattern: str,
              input_size: int,
              hidden_size: int,
              num_layers: int,
              val_once) -> Tuple[str, float]:
    """
    path_pattern ... e.g. '/workspace/checkpoints/gru_h*.pt'
    returns        ... (best_path, best_f1)
    """
    best_f1, best_path = -1., None
    for ckpt in glob(path_pattern):
        # build dummy backbone with the *same* arch as in the ckpt-file name
        model = GRUAnomalyDetector(input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers)
        sd = torch.load(ckpt, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)
        f1 = quick_f1(model, val_once, device="cpu")["F1"]
        print(f"{os.path.basename(ckpt):<22}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_path = f1, ckpt
    return best_path, best_f1
