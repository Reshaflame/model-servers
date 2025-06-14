# ── refresh_expected_features.py ───────────────────────────────────
import sys, os, glob, pandas as pd
from utils.meta_builder import save_feature_list
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.meta_builder import save_feature_list


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PHASE2_DIR = os.path.join(DATA_DIR, "chunks_labeled_phase2")

# 1️⃣  pick the first enhanced chunk
sample = sorted(glob.glob(os.path.join(PHASE2_DIR, "chunk_*_labeled_enc.csv")))[0]

# 2️⃣  read just the header to get column order
cols = pd.read_csv(sample, nrows=1).columns.tolist()

# 3️⃣  overwrite meta file
save_feature_list(cols)
print(f"✅ expected_features.json updated with {len(cols)} columns.")
