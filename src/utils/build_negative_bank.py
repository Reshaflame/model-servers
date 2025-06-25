import os, glob, numpy as np, pandas as pd, torch

SRC = "/workspace/model-servers/data/labeled_chunks"
OUT = "/workspace/model-servers/data/negative_bank.pt"
SEQ = 1      # same as GRU

xs = []
for f in glob.glob(os.path.join(SRC, "*.csv")):
    df   = pd.read_csv(f)
    mask = df["label"] == 0
    idxs = np.where(mask.values)[0]
    for i in idxs:
        if i < SEQ - 1:                # need full sequence
            continue
        seq = (df.iloc[i-SEQ+1:i+1]
                 .drop(columns="label")
                 .values.astype("float32"))
        xs.append(seq)

X = torch.tensor(np.asarray(xs))
torch.save({"X": X}, OUT)
print("💾 negative bank saved:", OUT, X.shape)
