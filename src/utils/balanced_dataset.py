# ─── src/utils/balanced_dataset.py ───────────────────────────────────
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

MAX_CATEGORIES = 16_000_000         # ≤ 2^24 – 1  (PyTorch multinomial limit)


class GlobalBalancedDataset(Dataset):
    """
    Dataset spanning *all* CSV chunks.
    Each sample is a (sequence, label) pair.  A weighted sampler
    is attached **only if** the total number of samples fits under
    PyTorch's multinomial hard-cap.  Otherwise we fall back to plain
    shuffling and rely on `pos_weight` inside the loss function.
    """

    def __init__(
        self,
        chunk_dir: str,
        feature_cols,
        label_col: str = "label",
        sequence_length: int = 10,
        minority_factor: float = 0.30,
        device: str = "cpu",
    ):
        self.seq_len = sequence_length
        self.label_col = label_col
        self.device = device
        self.chunk_dir = chunk_dir
        self.chunks = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".csv"))
        self.features = list(feature_cols)

        # ── Build global index of (chunk_id, row_id) -----------------
        row_ptrs, labels = [], []
        for cid, fname in enumerate(self.chunks):
            df = pd.read_csv(
                os.path.join(chunk_dir, fname), usecols=self.features + [label_col]
            )
            y = (df[label_col] == 1).astype(int).values
            for rid in range(len(df) - self.seq_len):
                row_ptrs.append((cid, rid))
                labels.append(y[rid + self.seq_len - 1])

        self.index = np.asarray(row_ptrs, dtype=np.int32)
        self.labels = np.asarray(labels, dtype=np.int8)

        # ── Build sampler only if within safe size -------------------
        n = len(self.labels)
        n_pos = int(self.labels.sum())

        if n <= MAX_CATEGORIES:
            w_pos = minority_factor / max(1, n_pos)
            w_neg = (1.0 - minority_factor) / (n - n_pos)
            weights = np.where(self.labels == 1, w_pos, w_neg)
            self.sampler = WeightedRandomSampler(
                weights, num_samples=n, replacement=True
            )
            print(
                f"[Sampler] ✔︎ Using WeightedRandomSampler "
                f"(samples={n:,}, positives={n_pos})"
            )
        else:
            # fall back – caller should pass shuffle=True in DataLoader
            self.sampler = None
            print(
                f"[Sampler] ⚠️  Dataset too large ({n:,} samples); "
                "sampler disabled – rely on pos_weight in loss."
            )

    # -----------------------------------------------------------------
    # PyTorch Dataset API
    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cid, rid = self.index[idx]
        df = pd.read_csv(
            os.path.join(self.chunk_dir, self.chunks[cid]),
            usecols=self.features + [self.label_col],
        )

        feats = (
            df[self.features]
            .values[rid : rid + self.seq_len]
            .astype(np.float32)
        )
        label = float(df[self.label_col].iloc[rid + self.seq_len - 1] == 1)

        X = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)
        if self.device != "cpu":
            X = X.to(self.device)
            y = y.to(self.device)
        return X, y
