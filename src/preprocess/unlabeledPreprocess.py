import os
import pandas as pd
import json

try:
    import cudf
    CUDF_AVAILABLE = False if os.getenv("CUDF_FORCE_DISABLE") == "true" else True
except ImportError:
    CUDF_AVAILABLE = False

# === Load known AUTH categories ===
with open("data/auth_categories.json") as f:
    AUTH_CATEGORIES = json.load(f)

def preprocess_unlabeled_chunk(chunk_path, all_categories, output_dir, chunk_id,
                                user_counter, comp_counter, batch_size=50000):
    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_preprocessed.csv")

    if os.path.exists(output_file):
        print(f"[Chunk {chunk_id}] ⏭️ Already processed, skipping.")
        return

    print(f"[Chunk {chunk_id}] 🚀 Reading {chunk_path}")
    df = pd.read_csv(chunk_path)

    if "time" not in df.columns:
        print(f"[Chunk {chunk_id}] ⚠️ Missing 'time' column, skipping.")
        return

    df = df.dropna(subset=["time"])
    all_batches = []

    get_user_freq = lambda u: user_counter.get(u, 0)
    get_comp_freq = lambda c: comp_counter.get(c, 0)

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size].copy()
        print(f"[Chunk {chunk_id}] 🧪 Rows {start}-{start + batch_size} (cuDF={CUDF_AVAILABLE})")

        try:
            # === Add frequency-encoded features ===
            batch_df["src_user_freq"] = batch_df["src_user"].map(get_user_freq).fillna(0)
            batch_df["dst_user_freq"] = batch_df["dst_user"].map(get_user_freq).fillna(0)
            batch_df["src_comp_freq"] = batch_df["src_comp"].map(get_comp_freq).fillna(0)
            batch_df["dst_comp_freq"] = batch_df["dst_comp"].map(get_comp_freq).fillna(0)

            if CUDF_AVAILABLE:
                batch = cudf.from_pandas(batch_df)
                for col in all_categories:
                    if col in batch.columns:
                        batch[col] = batch[col].astype(str)
                        allowed = set(all_categories[col])
                        batch[col] = batch[col].where(batch[col].isin(allowed), other='OTHER')
                        batch[col] = batch[col].astype('category').cat.set_categories(all_categories[col] + ["OTHER"])
                batch = cudf.get_dummies(batch, columns=all_categories.keys(), dummy_na=False).to_pandas()
            else:
                for col in all_categories:
                    if col in batch_df.columns:
                        batch_df[col] = batch_df[col].astype(str)
                        allowed = set(all_categories[col])
                        batch_df[col] = batch_df[col].where(batch_df[col].isin(allowed), other='OTHER')
                        batch_df[col] = pd.Categorical(batch_df[col], categories=all_categories[col] + ["OTHER"])
                batch = pd.get_dummies(batch_df, columns=list(all_categories.keys()), dummy_na=False)

            # === Ensure all dummy columns exist ===
            for col in all_categories:
                for cat in all_categories[col]:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in batch.columns:
                        batch[dummy_col] = 0

            all_batches.append(batch)
        except Exception as e:
            print(f"[❌ Chunk {chunk_id}, Batch {start}] Exception: {e}")

    if not all_batches:
        print(f"[Chunk {chunk_id}] ⚠️ No batches processed.")
        return

    chunk = pd.concat(all_batches, ignore_index=True)
    chunk.fillna(0, inplace=True)
    chunk.to_csv(output_file, index=False)
    print(f"[Chunk {chunk_id}] ✅ Saved to {output_file}")

def preprocess_auth_data(chunk_dir='data/shared_chunks'):
    print("[Preprocess] 🌍 Loading frequency maps...")
    user_freq_path = "data/user_freq.json"
    comp_freq_path = "data/comp_freq.json"

    if not os.path.exists(user_freq_path) or not os.path.exists(comp_freq_path):
        raise FileNotFoundError("Frequency maps not found. Please run labeled preprocessing first.")

    with open(user_freq_path, "r") as f:
        user_counter = json.load(f)
    with open(comp_freq_path, "r") as f:
        comp_counter = json.load(f)

    chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")])
    preprocessed_dir = "data/preprocessed_unlabeled/chunks"
    os.makedirs(preprocessed_dir, exist_ok=True)

    for i, chunk_path in enumerate(chunk_paths):
        preprocess_unlabeled_chunk(
            chunk_path=chunk_path,
            all_categories=AUTH_CATEGORIES,
            output_dir=preprocessed_dir,
            chunk_id=i,
            user_counter=user_counter,
            comp_counter=comp_counter
        )

    print(f"[Preprocess] ✅ {len(chunk_paths)} unlabeled chunks saved to: {preprocessed_dir}")

if __name__ == "__main__":
    print("[Preprocess] 🚀 Starting unlabeled preprocessing...")
    preprocess_auth_data()
    print("[Preprocess] 🏁 Done.")
