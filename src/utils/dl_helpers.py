# utils/dl_helpers.py
import torch

def stack_batch(batch):
    """Pure CPU collation – returns two stacked CPU tensors."""
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def to_device(batch, device, non_blocking=True):
    """Move an (x,y) batch that is *already* stacked to the given device."""
    x, y = batch
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
