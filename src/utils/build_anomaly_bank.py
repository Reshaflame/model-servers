# src/utils/build_anomaly_bank.py
import os, pandas as pd, numpy as np, torch, glob

SRC  = "/data/labeled_chunks"
OUT  = "/data/anomaly_bank.pt"
SEQ  = 10                       # use 1 for GRU

xs, ys = [], []
for fname in glob.glob(os.path.join(SRC, "*.csv")):
    df = pd.read_csv(fname)
    pos_idx = np.where(df["label"].values == 1)[0]
    for idx in pos_idx:
        if idx < SEQ-1: continue
        seq = df.iloc[idx-SEQ+1:idx+1].drop(columns="label").values
        xs.append(seq.astype("float32"))
        ys.append(1.)
xs, ys = torch.tensor(xs), torch.tensor(ys).unsqueeze(1)
torch.save({"X":xs, "y":ys}, OUT)
print(f"💾 anomaly bank: {xs.shape[0]} sequences → {OUT}")
