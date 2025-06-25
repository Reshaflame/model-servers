# ─── utils/build_anomaly_bank.py ─────────────────────────────────────
import os, glob, torch, pandas as pd, numpy as np

def _collect_numeric(df, feature_cols):
    """Return an (seq_len, F) float32 numpy array from df[feature_cols]."""
    return (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")  # strings → NaN
        .fillna(0.0)
        .values
        .astype("float32")
    )

def build_bank(src_dir: str, out_pt: str, feature_cols, seq_len: int = 10) -> None:
    xs, ys = [], []
    csvs = glob.glob(os.path.join(src_dir, "*.csv"))
    assert csvs, f"No CSV files found in {src_dir!r}"

    for fname in csvs:
        df = pd.read_csv(fname, usecols=feature_cols + ["label"])
        pos_idx = np.where(df["label"].values == 1)[0]
        for idx in pos_idx:
            if idx < seq_len - 1:
                continue
            window = _collect_numeric(
                df.iloc[idx - seq_len + 1 : idx + 1], feature_cols
            )
            xs.append(window)
            ys.append([1.0])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save({"X": xs, "y": ys}, out_pt)
    print(f"✅  anomaly bank built: {xs.shape[0]:,} sequences → {out_pt}")

def build_if_needed(src_dir: str, out_pt: str, feature_cols, seq_len: int = 10):
    """Re-build the bank iff it is missing or empty/corrupted."""
    if os.path.exists(out_pt):
        try:
            bank = torch.load(out_pt, map_location="cpu")
            if bank["X"].numel() > 0:
                return                        # ✓ ready to use
        except Exception:
            pass
        print("⚠️  anomaly_bank.pt is empty / corrupted – rebuilding …")

    build_bank(src_dir, out_pt, feature_cols, seq_len)

# --------------------------------------------------------------------
if __name__ == "__main__":                     # manual run
    SRC = "/data/labeled_chunks"
    OUT = "/data/anomaly_bank.pt"
    FEATS = (
        pd.read_csv(glob.glob(os.path.join(SRC, "*.csv"))[0])
          .drop(columns="label")
          .select_dtypes("number")
          .columns
          .tolist()
    )
    build_bank(SRC, OUT, FEATS, seq_len=10)
