# src/utils/dl_helpers.py
import torch

def to_device(batch, device):
    """Batch = list[(x, y)].  Returns tuple(tensor[B,…], tensor[B,1])."""
    xs, ys = zip(*batch)             # list → tuple of tensors
    return (torch.stack(xs).to(device, non_blocking=True),
            torch.stack(ys).to(device, non_blocking=True))
