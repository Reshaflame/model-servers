# src/utils/refresh_expected_features.py

import os, sys, glob
import pandas as pd

# Add project root to sys.path (for direct script execution)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.meta_builder import save_feature_list  # robust import

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PHASE2_DIR = os.path.join(DATA_DIR, "chunks_labeled_phase2")

def refresh_expected_features():
    chunks = sorted(glob.glob(os.path.join(PHASE2_DIR, "chunk_*_labeled_enc.csv")))
    if not chunks:
        print("❌ No enhanced chunks found.")
        return

    sample = chunks[0]
    cols = pd.read_csv(sample, nrows=1).columns.tolist()
    save_feature_list(cols)
    print(f"✅ expected_features.json updated with {len(cols)} columns.")

if __name__ == "__main__":
    print("[Meta] ♻️ Refreshing expected features from phase 2 enhanced chunks...")
    refresh_expected_features()
