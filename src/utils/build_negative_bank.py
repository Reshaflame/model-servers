# ─── src/utils/build_negative_bank.py ──────────────────────────────
"""
Creates / refreshes a bank of **negative** sequences so FastBalancedDS
can draw extra negatives straight from memory.

One-off usage:
    python -m src.utils.build_negative_bank
or  python -m src.utils.build_negative_bank --seq 1 --src <dir> --out <file>
"""
from __future__ import annotations
import os, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------ #
#  Constants                                                         #
# ------------------------------------------------------------------ #
try:                                 # when executed as module
    from .constants import CHUNKS_LABELED_PATH
except ImportError:                  # when imported from elsewhere
    from utils.constants import CHUNKS_LABELED_PATH

# ------------------------------------------------------------------ #
#  Core builder                                                      #
# ------------------------------------------------------------------ #
def build_bank(src_dir: str,
               out_pt: str,
               seq_len: int = 1,
               feature_cols: list[str] | None = None) -> None:
    """Read *all* CSVs in `src_dir`, keep only negatives, save a .pt."""
    src_dir = Path(src_dir)
    files   = sorted(src_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {src_dir}")

    # Determine numeric feature columns once
    if feature_cols is None:
        sample = pd.read_csv(files[0])
        feature_cols = [c for c in sample.select_dtypes("number").columns
                        if c != "label"]

    xs = []
    for f in files:
        df   = pd.read_csv(f)
        negs = np.where(df["label"].values == 0)[0]
        for i in negs:
            if i < seq_len - 1:
                continue
            # -------- Phase-2 robust conversion ---------------------------------
            chunk = df.iloc[i-seq_len+1:i+1][feature_cols]

            # add missing dummy columns
            for col in feature_cols:
                if col not in chunk.columns:
                    chunk[col] = 0.0

            # coerce strings → numeric → float32
            seq = (chunk
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .astype("float32")
                    .values)
            # --------------------------------------------------------------------
            xs.append(seq)

    if not xs:
        raise RuntimeError("No negative sequences found — bank would be empty.")

    X = torch.tensor(np.asarray(xs))          # (N, seq_len, F)
    os.makedirs(Path(out_pt).parent, exist_ok=True)
    torch.save({"X": X}, out_pt)
    with open(str(out_pt) + ".meta", "w") as f:
        f.write(str(X.shape))  # e.g., (17180217, 1, 56)
    print(f"✅  negative bank built: {X.shape[0]} sequences → {out_pt}")

# ------------------------------------------------------------------ #
#  Helper: build only when required                                  #
# ------------------------------------------------------------------ #
def build_if_needed(src_dir: str,
                    out_pt: str,
                    feature_cols: list[str],
                    seq_len: int = 1) -> None:
    """
    • If `out_pt` exists and `.meta` file matches shape (seq_len, |F|),
      skip rebuilding.
    • Otherwise rebuild the anomaly bank with `build_bank(...)`.
    """
    meta_path = out_pt + ".meta"

    if os.path.exists(out_pt) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = f.read().strip().split(",")
                saved_seq_len = int(meta[0])
                saved_num_feat = int(meta[1])
                num_samples = int(meta[2])
            
            if saved_seq_len == seq_len and saved_num_feat == len(feature_cols):
                print(f"ℹ️  {out_pt} already exists with {num_samples} sequences — nothing to do.")
                return
            else:
                print(f"⚠️  {out_pt} exists but is incompatible (meta mismatch) — rebuilding …")
        except Exception as e:
            print(f"⚠️  meta file unreadable or corrupt ({e}) — rebuilding …")

    # otherwise: rebuild and write meta
    build_bank(src_dir,
           out_pt,
           seq_len=seq_len,
           feature_cols=feature_cols)
    try:
        bank = torch.load(out_pt, map_location="cpu")
        shape = bank["X"].shape  # (N, seq, F)
        with open(meta_path, "w") as f:
            f.write(f"{shape[1]},{shape[2]},{shape[0]}")  # seq_len,num_feat,num_samples
    except Exception as e:
        print(f"⚠️  Failed to write meta after rebuild: {e}")


# ------------------------------------------------------------------ #
#  CLI entry-point                                                   #
# ------------------------------------------------------------------ #
def main() -> None:
    root = Path(__file__).resolve().parents[2]      # project root
    default_src = root / CHUNKS_LABELED_PATH
    default_out = root / "data" / "negative_bank.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(default_src),
                    help="folder with labelled chunk CSVs")
    ap.add_argument("--out", default=str(default_out),
                    help="output .pt file")
    ap.add_argument("--seq", type=int, default=1,
                    help="sequence length (1 for GRU, 10 for LSTM …)")
    args = ap.parse_args()

    build_bank(args.src, args.out, seq_len=args.seq)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
