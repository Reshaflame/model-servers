# ─── src/utils/build_anomaly_bank.py ─────────────────────────────
"""
Builds / refreshes an anomaly-only sequence bank so we can draw
positives quickly during training.

Usage (one-off):
    python -m src.utils.build_anomaly_bank         # <── default paths
or   python -m src.utils.build_anomaly_bank --seq 1  --src custom_dir --out custom_bank.pt
"""
from __future__ import annotations
import os, argparse, glob, numpy as np, pandas as pd, torch
from pathlib import Path

try:                                   
    from .constants import CHUNKS_LABELED_PATH
except ImportError:                    
    from utils.constants import CHUNKS_LABELED_PATH

# -----------------------------------------------------------------
def build_bank(src_dir: str, out_pt: str, seq_len: int = 1,
               feature_cols: list[str] | None = None) -> None:
    src_dir = Path(src_dir)
    files   = sorted(src_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {src_dir}")

    # --- decide which numeric columns to keep (once) --------------
    if feature_cols is None:
        sample_df   = pd.read_csv(files[0])
        feature_cols = [c for c in sample_df.select_dtypes("number").columns
                        if c != "label"]

    xs, ys = [], []
    for f in files:
        df  = pd.read_csv(f)
        pos = np.where(df["label"].values == 1)[0]
        for i in pos:
            if i < seq_len - 1:
                continue
            seq = (df.iloc[i-seq_len+1:i+1][feature_cols]
                     .values.astype("float32"))
            xs.append(seq)
            ys.append(1.)

    if not xs:
        raise RuntimeError("No positive sequences found — bank would be empty.")

    X = torch.tensor(np.asarray(xs))          # (P, seq, F)
    y = torch.ones(len(xs), 1)                # all ones

    torch.save({"X": X, "y": y}, out_pt)
    print(f"✅  anomaly bank built: {X.shape[0]} sequences → {out_pt}")


# -----------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parents[2]      # repo root
    default_src = root / CHUNKS_LABELED_PATH
    default_out = root / "data" / "anomaly_bank.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  default=str(default_src),
                    help="folder with labeled chunk CSVs")
    ap.add_argument("--out",  default=str(default_out),
                    help="output .pt file")
    ap.add_argument("--seq",  type=int, default=1,
                    help="sequence length (1 for GRU, 10 for LSTM, etc.)")
    args = ap.parse_args()

    out_pt = Path(args.out)
    if out_pt.exists():
        try:
            bank = torch.load(out_pt, map_location="cpu")
            if "X" in bank and bank["X"].numel() > 0:
                print(f"ℹ️  {out_pt} already exists with "
                      f"{bank['X'].shape[0]} sequences — nothing to do.")
                return
        except Exception:
            print("⚠️  existing file unreadable — rebuilding …")

    build_bank(args.src, out_pt, seq_len=args.seq)

# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
