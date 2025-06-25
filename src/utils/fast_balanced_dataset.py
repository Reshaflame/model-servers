# utils/fast_balanced_dataset.py
import torch, pandas as pd, os, random, glob
from torch.utils.data import Dataset

class FastBalancedDS(Dataset):
    def __init__(self,
                 chunk_dir,
                 bank_pt,
                 feature_cols,
                 seq_len=10,
                 pos_ratio=0.30):
        bank = torch.load(bank_pt, map_location="cpu")
        self.pos_X = bank["X"]          # (P, seq, F)  — all positives
        self.pos_y = bank["y"]          # (P, 1)

        self.neg_files = glob.glob(os.path.join(chunk_dir, "*.csv"))
        self.cols   = feature_cols
        self.seq    = seq_len
        self.ratio  = pos_ratio

    # ------------------------------------------------------------
    # NEW: quick helper for one negative sample (used by pipeline)
    # ------------------------------------------------------------
    def sample_negative(self):
        """
        Draw a **single negative** (X, y) tuple as CPU tensors.
        Guaranteed label == 0.
        """
        while True:
            f = random.choice(self.neg_files)
            df = pd.read_csv(f, usecols=self.cols + ["label"])
            start = random.randrange(self.seq, len(df))
            if df["label"].iloc[start] == 1:      # skip positives
                continue
            seq = df[self.cols].iloc[start-self.seq:start].values.astype("float32")
            X = torch.tensor(seq)                 # (seq, F)
            y = torch.tensor([0.], dtype=torch.float32)
            return X, y

    # ------------------------------------------------------------
    # standard Dataset methods (unchanged)
    # ------------------------------------------------------------
    def __len__(self):          # synthetic — large enough
        return 100_000_000

    def __getitem__(self, _):
        if random.random() < self.ratio:
            i = random.randrange(len(self.pos_X))
            return self.pos_X[i], self.pos_y[i]

        # otherwise draw a negative on-the-fly
        return self.sample_negative()
