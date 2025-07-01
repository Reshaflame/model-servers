import torch, pandas as pd, os, glob, random
from torch.utils.data import Dataset

class FastBalancedDS(Dataset):
    """
    All positives & negatives are pre-materialised tensors → no CSV I/O
    """

    def __init__(self,
                 chunk_dir: str,
                 bank_pt: str,
                 neg_bank_pt: str,
                 feature_cols,
                 seq_len: int = 1,
                 pos_ratio: float = 0.30):

        bank = torch.load(bank_pt, map_location="cpu")
        self.pos_X = bank["X"]          # (P, seq, F)
        self.pos_y = bank["y"]          # (P, 1)

        neg_bank = torch.load(neg_bank_pt, map_location="cpu")
        self.neg_X = neg_bank["X"]      # (N, seq, F)

        # ── only needed for sample_negative() fallback -------------
        self.neg_files = glob.glob(os.path.join(chunk_dir, "*.csv"))

        self.cols   = list(feature_cols)    # ← 61 columns from expected_features.json
        self.seq    = seq_len
        self.ratio  = pos_ratio
        self._cache = {}

    # -------------------------------------------------------------- #
    def __len__(self):                     # synthetic – huge
        return 100_000_000

    def __getitem__(self, _):
        if random.random() < self.ratio:          # positive
            i = random.randrange(len(self.pos_X))
            return self.pos_X[i], self.pos_y[i]
        else:                                     # negative
            j = random.randrange(len(self.neg_X))
            return self.neg_X[j], torch.tensor([0.], dtype=torch.float32)

    # ---------- OPTIONAL: fallback CSV sampler -------------------- #
    def _load_csv(self, path):
        if path not in self._cache:
            df = pd.read_csv(path, usecols=self.cols + ["label"])

            # ── PATCH ➊: add missing dummies & coerce numeric ──────────
            for col in self.cols:
                if col not in df.columns:
                    df[col] = 0.0                      # add all-zero dummy

            df[self.cols] = (
                df[self.cols]
                  .apply(pd.to_numeric, errors="coerce")
                  .fillna(0.0)
                  .astype("float32")
            )
            # ───────────────────────────────────────────────────────────

            self._cache[path] = df
            if len(self._cache) > 10:                 # keep ≤10 dfs
                self._cache.pop(next(iter(self._cache)))
        return self._cache[path]

    def sample_negative(self):
        """Rarely used helper for quick stats in the pipeline."""
        while True:
            f   = random.choice(self.neg_files)
            df  = self._load_csv(f)
            idx = random.randrange(self.seq, len(df))
            if df["label"].iloc[idx] == 1:            # skip positives
                continue

            # ── PATCH ➋: guaranteed 61-features seq tensor ────────────
            seq = df[self.cols].iloc[idx-self.seq:idx].values
            x   = torch.tensor(seq, dtype=torch.float32)
            # ───────────────────────────────────────────────────────────

            y   = torch.tensor([0.], dtype=torch.float32)
            return x, y
