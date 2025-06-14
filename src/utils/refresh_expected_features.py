# ── refresh_expected_features.py ───────────────────────────────────
import sys, os, glob, pandas as pd

# ✅ Fix sys.path BEFORE any internal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.meta_builder import save_feature_list

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PHASE2_DIR = os.path.join(DATA_DIR, "chunks_labeled_phase2")

# 1️⃣ Pick the first enhanced chunk
sample = sorted(glob.glob(os.path.join(PHASE2_DIR, "chunk_*_labeled_enc.csv")))[0]

# 2️⃣ Read just the header to get column order
cols = pd.read_csv(sample, nrows=1).columns.tolist()

# 3️⃣ Overwrite meta file
save_feature_list(cols)
print(f"✅ expected_features.json updated with {len(cols)} columns.")
