"""
Phase-2 enhancer: one-hot-encode auth_type, logon_type, auth_orientation
for every already-preprocessed labeled chunk.

Input  dir : data/chunks_labeled/*.csv     (Phase 1 output)
Output dir : data/chunks_labeled_phase2/   (new)
Category map: data/auth_categories.json    (already committed)
"""

import os, json, glob, pandas as pd, numpy as np
from utils.chunk_archiver import zip_chunks    # reuse existing archiver

ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
SRC_DIR   = os.path.join(DATA_DIR, "chunks_labeled")
DST_DIR   = os.path.join(DATA_DIR, "chunks_labeled_phase2")
os.makedirs(DST_DIR, exist_ok=True)

# ── load canonical category lists ──────────────────────────────────
with open(os.path.join(DATA_DIR, "auth_categories.json"), "r") as f:
    CAT_MAP = json.load(f)

CAT_COLUMNS = ["auth_type", "logon_type", "auth_orientation"]

def enhance_chunk(src_path: str, dst_path: str):
    df = pd.read_csv(src_path)
    # one-hot encode using full category set to keep column order stable
    df = pd.get_dummies(
        df,
        columns=CAT_COLUMNS,
        dtype=np.float32,
        prefix=CAT_COLUMNS,
        prefix_sep="_",
    )
    # ensure every expected dummy column exists
    for col in CAT_COLUMNS:
        for cat in CAT_MAP[col]:
            dummy = f"{col}_{cat}"
            if dummy not in df.columns:
                df[dummy] = 0.0
    # keep column order deterministic
    fixed_order = [
        c for c in df.columns
        if not any(c.startswith(f"{p}_") for p in CAT_COLUMNS)
    ]
    for col in CAT_COLUMNS:
        fixed_order += [f"{col}_{cat}" for cat in CAT_MAP[col]]
    df = df[fixed_order]
    df.to_csv(dst_path, index=False)

def run_phase2():
    chunk_files = sorted(glob.glob(os.path.join(SRC_DIR, "chunk_*_labeled_feat.csv")))
    processed = 0
    for src in chunk_files:
        fname = os.path.basename(src)
        dst   = os.path.join(DST_DIR, fname.replace("_feat", "_enc"))
        if os.path.exists(dst):
            print(f"[{fname}] ⏭️  Skipped (already enhanced)")
            continue
        print(f"[{fname}] 🚀 Enhancing …")
        enhance_chunk(src, dst)
        processed += 1
    print(f"[Phase-2] ✅ Enhanced {processed} chunks.")
    
    # --- refresh expected_features.json ---------------------------------
    sample_chunk = sorted(glob.glob(os.path.join(DST_DIR, "*.csv")))[0]
    cols = pd.read_csv(sample_chunk, nrows=1).columns.tolist()
    from utils.meta_builder import save_feature_list
    save_feature_list(cols)
    print("[Phase-2] 📝 expected_features.json updated.")

    # optional: zip the new directory for easy download
    zip_chunks(DST_DIR, kind="labeled_phase2")

if __name__ == "__main__":
    run_phase2()
