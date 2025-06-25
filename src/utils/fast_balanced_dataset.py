# utils/fast_balanced_dataset.py
import torch, pandas as pd, os, random, glob
from torch.utils.data import Dataset

class FastBalancedDS(Dataset):
    def __init__(self, chunk_dir, bank_pt, feature_cols, seq_len=10,
                 pos_ratio=0.3):
        bank = torch.load(bank_pt, map_location="cpu")
        self.pos_X = bank["X"];  self.pos_y = bank["y"]          # tensors

        self.neg_files = glob.glob(os.path.join(chunk_dir, "*.csv"))
        self.cols  = feature_cols
        self.seq   = seq_len
        self.ratio = pos_ratio
        self.device = None      # keep cpu; we move in collate_fn

    def __len__(self):               
        return 200_000

    def __getitem__(self, _):
        if random.random() < self.ratio:
            i = random.randrange(len(self.pos_X))
            return self.pos_X[i], self.pos_y[i]
        # —— sample a random negative sequence —————————————
        while True:
            f = random.choice(self.neg_files)
            df = pd.read_csv(f, usecols=self.cols+["label"])
            start = random.randrange(self.seq, len(df))
            if df["label"].iloc[start] == 1:          # avoid positives
                continue
            seq = df[self.cols].iloc[start-self.seq:start].values.astype("float32")
            return torch.tensor(seq), torch.tensor([0.], dtype=torch.float32)
