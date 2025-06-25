# ─── utils/build_anomaly_bank.py ────────────────────────────────────
import os, torch, glob, pandas as pd, numpy as np

def build_bank(src_dir: str, out_pt: str, seq_len: int = 10) -> None:
    """Re-builds anomaly bank (.pt) from every CSV in src_dir."""
    xs, ys = [], []
    csvs = glob.glob(os.path.join(src_dir, "*.csv"))
    assert csvs, f"No CSV files found in {src_dir!r}"

    for fname in csvs:
        df = pd.read_csv(fname)
        pos = np.where(df["label"].values == 1)[0]
        for i in pos:
            if i < seq_len - 1:
                continue
            window = (
                df.iloc[i - seq_len + 1 : i + 1]
                .drop(columns="label")
                .values.astype("float32")
            )
            xs.append(window)
            ys.append([1.0])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save({"X": xs, "y": ys}, out_pt)
    print(f"✅  anomaly bank built: {xs.shape[0]:,} sequences → {out_pt}")

def build_if_needed(src_dir: str, out_pt: str, seq_len: int = 10):
    if os.path.exists(out_pt):
        try:
            bank = torch.load(out_pt, map_location="cpu")
            if bank["X"].numel() > 0:
                return                              # ok
        except Exception:
            pass                                    # corrupted → rebuild
        print("⚠️  anomaly_bank.pt is empty / corrupted – rebuilding …")

    build_bank(src_dir, out_pt, seq_len)

# -------------------------------------------------------------------
if __name__ == "__main__":            # so you can also run `python …/build_anomaly_bank.py`
    SRC = "/data/labeled_chunks"
    OUT = "/data/anomaly_bank.pt"
    SEQ = 10                          # 1 for GRU, 10 for LSTM
    build_bank(SRC, OUT, SEQ)
