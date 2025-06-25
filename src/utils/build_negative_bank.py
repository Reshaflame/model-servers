# ─── src/utils/build_negative_bank.py ────────────────────────────
"""
Collects only *negative* sequences into a single .pt file so the
FastBalancedDS can draw “extra” negatives without touching the disk.

Usage (one-off):
    python -m src.utils.build_negative_bank        # default paths
or  python -m src.utils.build_negative_bank --seq 1 --src custom_dir --out custom_bank.pt
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:                                    # when run as  python -m src.utils.build_negative_bank
    from .constants import CHUNKS_LABELED_PATH
except ImportError:                     # when imported as  from utils.build_negative_bank import …
    from utils.constants import CHUNKS_LABELED_PATH
    
# -----------------------------------------------------------------
def build_bank(src_dir: str, out_pt: str, seq_len: int = 1,
               feature_cols: list[str] | None = None) -> None:
    src_dir = Path(src_dir)
    files   = sorted(src_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {src_dir}")

    # --- decide which numeric columns to use (once) ---------------
    if feature_cols is None:
        sample_df   = pd.read_csv(files[0])
        feature_cols = [c for c in sample_df.select_dtypes("number").columns
                        if c != "label"]

    xs = []
    for f in files:
        df   = pd.read_csv(f)
        negs = np.where(df["label"].values == 0)[0]
        for i in negs:
            if i < seq_len - 1:
                continue
            seq = (df.iloc[i-seq_len+1:i+1][feature_cols]
                     .values.astype("float32"))
            xs.append(seq)

    if not xs:
        raise RuntimeError("No negative sequences found — bank would be empty.")

    X = torch.tensor(np.asarray(xs))        # (N, seq, F)
    torch.save({"X": X}, out_pt)
    print(f"✅  negative bank built: {X.shape[0]} sequences → {out_pt}")

# -----------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parents[2]      # repo root
    default_src = root / CHUNKS_LABELED_PATH
    default_out = root / "data" / "negative_bank.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(default_src),
                    help="folder with labeled chunk CSVs")
    ap.add_argument("--out", default=str(default_out),
                    help="output .pt file")
    ap.add_argument("--seq", type=int, default=1,
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
