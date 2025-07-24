import numpy as np, json, itertools, argparse, os
from pathlib import Path

def streaming_f1(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum();  fp = ((y_true==0)&(y_pred==1)).sum()
    fn = ((y_true==1)&(y_pred==0)).sum()
    return 2*tp / (2*tp + fp + fn + 1e-9)

def main(pred_dir="models/preds", step=.05):
    y   = np.load(f"{pred_dir}/y_true.npy")

    p_g = np.load(f"{pred_dir}/gru_preds.npy")
    p_l = np.load(f"{pred_dir}/lstm_preds.npy")
    p_i = np.load(f"{pred_dir}/iso_preds.npy")

    best = (0., (1/3,1/3,1/3))
    for w1, w2 in itertools.product(np.arange(0,1+step,step), repeat=2):
        w3 = 1 - w1 - w2
        if w3 < 0:        # keep sum=1
            continue
        s = w1*p_g + w2*p_l + w3*p_i
        f1 = streaming_f1(y, (s >= 0.5).astype(int))
        if f1 > best[0]:
            best = (f1, (w1, w2, w3))

    print(f"🥇 Best F1={best[0]:.4f}  weights={best[1]}")
    Path("models").mkdir(exist_ok=True)
    with open("models/ensemble_weights.json","w") as f:
        json.dump(best[1], f)
