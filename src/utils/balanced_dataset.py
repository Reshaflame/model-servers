import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split

class GlobalBalancedDataset(Dataset):
    """
    One dataset spanning **all** chunk CSVs.
    We build an index (chunk_id, row_id) and sample with per-row weights
    so that positives appear in every mini-batch.
    """
    def __init__(self, chunk_dir, feature_cols, label_col="label",
                 sequence_length=10, minority_factor=0.3, device="cpu"):

        self.seq_len    = sequence_length
        self.label_col  = label_col
        self.device     = device
        self.chunks     = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".csv"))
        self.chunk_dir  = chunk_dir
        self.features   = feature_cols

        row_ptrs, labels = [], []        # build global index
        for cid, fname in enumerate(self.chunks):
            df = pd.read_csv(os.path.join(chunk_dir, fname),
                             usecols=self.features + [label_col])
            y  = (df[label_col] == 1).astype(int).values
            for rid in range(len(df) - self.seq_len):
                row_ptrs.append((cid, rid))
                labels.append(y[rid + self.seq_len - 1])

        self.index  = np.asarray(row_ptrs, dtype=np.int32)
        self.labels = np.asarray(labels,   dtype=np.int8)

        # ── weighted sampler so P(y=1) ≈ minority_factor ────────────
        n_pos = self.labels.sum(); n = len(self.labels)
        w_pos = minority_factor / max(1, n_pos)
        w_neg = (1 - minority_factor) / (n - n_pos)
        weights = np.where(self.labels == 1, w_pos, w_neg)
        self.sampler = WeightedRandomSampler(weights, num_samples=n, replacement=True)

    # -------- torchvision-style Dataset API --------------------------
    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        cid, rid = self.index[idx]
        df = pd.read_csv(os.path.join(self.chunk_dir, self.chunks[cid]),
                         usecols=self.features + [self.label_col])
        feats = df[self.features].values[rid : rid + self.seq_len].astype(np.float32)
        label = float(df[self.label_col].iloc[rid + self.seq_len - 1] == 1)
        X = torch.tensor(feats,  device=self.device)
        y = torch.tensor([label], device=self.device)
        return X, y
