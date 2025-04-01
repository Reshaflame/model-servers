import os
import pandas as pd
import torch
import shutil
import json

with open("data/auth_categories.json") as f:
    AUTH_CATEGORIES = json.load(f)

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

def preprocess_chunk(chunk_path, all_categories, output_dir):
    df = pd.read_csv(chunk_path)
    chunk_id = os.path.basename(chunk_path).split(".")[0]  # e.g., "chunk_0"

    print(f"[Preprocess] 🚀 Processing {chunk_id}...")

    if CUDF_AVAILABLE:
        try:
            chunk = cudf.from_pandas(df)
            chunk['time'] = cudf.to_datetime(chunk['time'], errors='coerce', unit='s')
            chunk = chunk.dropna(subset=['time'])
            for col in all_categories:
                chunk[col] = chunk[col].astype('category')
                chunk[col] = chunk[col].cat.set_categories(all_categories[col])
            chunk = cudf.get_dummies(chunk, columns=list(all_categories.keys()), dummy_na=False)
            chunk = chunk.fillna(0)
            chunk = chunk.to_pandas()
        except Exception as e:
            print(f"[cuDF Error] ❌ Falling back to pandas: {e}")
            CUDF_AVAILABLE = False  # fallback if error
    if not CUDF_AVAILABLE:
        try:
            df['time'] = pd.to_datetime(df['time'], errors='coerce', unit='s')
            df = df.dropna(subset=['time'])
            for col in all_categories:
                df[col] = pd.Categorical(df[col], categories=all_categories[col])
            df = pd.get_dummies(df, columns=list(all_categories.keys()), dummy_na=False)
            df = df.fillna(0)
            chunk = df
        except Exception as e:
            print(f"[Pandas Error] ❌ Failed to preprocess chunk {chunk_id}: {e}")
            return

    # Save chunk
    os.makedirs(output_dir, exist_ok=True)
    chunk_output = os.path.join(output_dir, f"{chunk_id}_preprocessed.csv")
    chunk.to_csv(chunk_output, index=False)
    print(f"✅ Saved preprocessed chunk to {chunk_output}")

    # # Copy for download access
    # downloadable_path = f"/app/models/{chunk_id}_unlabeled.csv"
    # shutil.copy(chunk_output, downloadable_path)
    # print(f"📦 Moved to download folder: {downloadable_path}")

def preprocess_auth_data(chunk_dir='data/shared_chunks'):
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")])

    from utils.constants import CHUNKS_SHARED_PATH

    all_categories = AUTH_CATEGORIES

    # Step 2: Process each chunk separately
    preprocessed_dir = "data/preprocessed_unlabeled/chunks"
    os.makedirs(preprocessed_dir, exist_ok=True)
    for chunk_path in chunk_paths:
        preprocess_chunk(chunk_path, all_categories, preprocessed_dir)

    print(f"✅ Preprocessed {len(chunk_paths)} unlabeled chunks saved to: {preprocessed_dir}")

if __name__ == "__main__":
    print("[Preprocess] Starting unlabeled preprocessing using shared chunks...")
    preprocess_auth_data()
    print("[Preprocess] Unlabeled preprocessing completed.")
