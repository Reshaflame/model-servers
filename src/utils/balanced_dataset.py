# ─── src/utils/balanced_dataset.py ──────────────────────────────────
import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, WeightedRandomSampler

MAX_CATEGORIES = 16_000_000                # multinomial hard-cap

class GlobalBalancedDataset(Dataset):
    """
    Global sequence dataset over all CSV chunks.

    • If #samples ≤ 16M →   WeightedRandomSampler with minority_factor.
    • Otherwise         →   plain shuffle; rely on BCEWithLogitsLoss(pos_weight).
    • Each chunk is read **once** into RAM (float32 / uint8), so per-sample
      access is ~zero-cost even with multi-worker loaders.
    """

    def __init__(self, chunk_dir: str,
                 feature_cols,
                 label_col: str = "label",
                 sequence_length: int = 10,
                 minority_factor: float = .30,
                 device: str = "cpu"):

        self.seq_len   = sequence_length
        self.device    = device
        self.label_col = label_col
        self.features  = list(feature_cols)

        # ------------------------------------------------------------
        # 1)  Load every chunk once, keep in RAM
        # ------------------------------------------------------------
        self._feat_arrays = []   # list[np.ndarray]  (rows, F)
        self._label_arrays = []  # list[np.ndarray]  (rows,)
        self.index = []          # list[(chunk_id, row_id)]
        chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".csv"))

        for cid, fname in enumerate(chunk_files):
            df = pd.read_csv(os.path.join(chunk_dir, fname),
                             usecols=self.features + [label_col])

            feat_mat = (df[self.features]
                        .to_numpy(dtype=np.float32, copy=True))
            lab_vec  = (df[label_col] == 1).to_numpy(dtype=np.uint8, copy=True)

            self._feat_arrays.append(feat_mat)
            self._label_arrays.append(lab_vec)

            # build global index (cid, start_row) for every valid sequence
            max_start = len(df) - sequence_length
            self.index.extend([(cid, rid) for rid in range(max_start)])

        self.index  = np.asarray(self.index,  dtype=np.int32)
        self.labels = np.asarray([self._label_arrays[cid][rid + sequence_length - 1]
                                  for cid, rid in self.index],
                                 dtype=np.uint8)

        # ------------------------------------------------------------
        # 2)  Sampler (only if small enough)
        # ------------------------------------------------------------
        n_samples = len(self.labels)
        n_pos     = int(self.labels.sum())

        if n_samples <= MAX_CATEGORIES:
            w_pos = minority_factor / max(1, n_pos)
            w_neg = (1.0 - minority_factor) / (n_samples - n_pos)
            weights = np.where(self.labels == 1, w_pos, w_neg)
            self.sampler = WeightedRandomSampler(weights,
                                                 num_samples=n_samples,
                                                 replacement=True)
            print(f"[Sampler] ✔︎ WeightedRandomSampler activated "
                  f"(samples={n_samples:,}, positives={n_pos})")
        else:
            self.sampler = None
            print(f"[Sampler] ⚠︎ Dataset too large ({n_samples:,}); "
                  "sampler disabled – pos_weight in loss will be used.")

    # ----------------------------------------------------------------
    # PyTorch Dataset API
    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cid, rid = self.index[idx]
        feats  = self._feat_arrays [cid][rid : rid + self.seq_len]
        label  = self._label_arrays[cid][rid + self.seq_len - 1]

        X = torch.from_numpy(feats)                  # float32
        y = torch.tensor([float(label)], dtype=torch.float32)

        if self.device != "cpu":
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return X, y
