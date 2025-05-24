import os
import pandas as pd
import gzip
import json
from collections import Counter

with open("data/auth_categories.json") as f:
    AUTH_CATEGORIES = json.load(f)

try:
    import cudf
    CUDF_AVAILABLE = False if os.getenv("CUDF_FORCE_DISABLE") == "true" else True
except ImportError:
    CUDF_AVAILABLE = False

import json

def process_labeled_chunk(chunk_path, redteam_events, categorical_columns, output_dir, chunk_id,
                          user_counter, comp_counter, batch_size=50000):
    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_labeled.csv")
    user_freq_path = "data/user_freq.json"
    comp_freq_path = "data/comp_freq.json"

    # Load cached frequency maps if they exist
    if os.path.exists(user_freq_path) and os.path.exists(comp_freq_path):
        with open(user_freq_path, "r") as f:
            user_counter = json.load(f)
        with open(comp_freq_path, "r") as f:
            comp_counter = json.load(f)

    if os.path.exists(output_file):
        print(f"[Chunk {chunk_id}] ⏭️ Already processed, skipping.")
        return

    print(f"[Chunk {chunk_id}] 🔍 Reading {chunk_path}")
    df = pd.read_csv(chunk_path)

    if "time" not in df.columns:
        print(f"[Chunk {chunk_id}] ⚠️ Missing 'time' column, skipping.")
        return

    df = df.dropna(subset=["time"])
    all_batches = []

    # Frequency mapping functions
    get_user_freq = lambda u: user_counter.get(u, 0)
    get_comp_freq = lambda c: comp_counter.get(c, 0)

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size].copy()
        print(f"[Chunk {chunk_id}] 🧪 Processing rows {start} to {start + batch_size} (cuDF={CUDF_AVAILABLE})")
    
        try:
            # print(f"[DEBUG] 🔍 Columns BEFORE encoding: {list(batch_df.columns)}")
        
            # Frequency-encode user/machine fields
            batch_df["src_user_freq"] = batch_df["src_user"].map(get_user_freq).fillna(0)
            batch_df["dst_user_freq"] = batch_df["dst_user"].map(get_user_freq).fillna(0)
            batch_df["src_comp_freq"] = batch_df["src_comp"].map(get_comp_freq).fillna(0)
            batch_df["dst_comp_freq"] = batch_df["dst_comp"].map(get_comp_freq).fillna(0)
        
            if CUDF_AVAILABLE:
                batch = cudf.from_pandas(batch_df)
                for col in categorical_columns:
                    if col in batch.columns:
                        batch[col] = batch[col].astype(str)
                        allowed = set(AUTH_CATEGORIES[col])
                        batch[col] = batch[col].where(batch[col].isin(allowed), other='OTHER')
                        batch[col] = batch[col].astype('category').cat.set_categories(AUTH_CATEGORIES[col] + ["OTHER"])
                batch = cudf.get_dummies(batch, columns=categorical_columns, dummy_na=False)
                batch = batch.to_pandas()
            else:
                print(f"[Chunk {chunk_id} | Batch {start}-{start+batch_size}] ⚠️ cuDF not available. Using pandas only.")
                for col in categorical_columns:
                    if col not in batch_df.columns:
                        # print(f"[DEBUG] ⛔ Column '{col}' missing in batch_df before encoding.")
                        continue
                
                    # print(f"[DEBUG] ✅ Processing categorical column: {col}")
                    batch_df[col] = batch_df[col].astype(str)
                
                    # Only apply category limiting if we have predefined categories
                    if col in AUTH_CATEGORIES:
                        allowed = set(AUTH_CATEGORIES[col])
                        batch_df[col] = batch_df[col].where(batch_df[col].isin(allowed), other='OTHER')
                        batch_df[col] = pd.Categorical(batch_df[col], categories=AUTH_CATEGORIES[col] + ["OTHER"])

        
                # print(f"[DEBUG] 🧱 Columns just before get_dummies: {list(batch_df.columns)}")
                batch = pd.get_dummies(batch_df, columns=[col for col in categorical_columns if col in batch_df.columns], dummy_na=False)
                # print(f"[DEBUG] 🧱 Columns AFTER get_dummies: {list(batch.columns)}")
        
            # Ensure consistent dummy columns
            for col in categorical_columns:
                if col not in AUTH_CATEGORIES:
                    continue  # Skip consistency check if no defined categories
                for cat in AUTH_CATEGORIES[col]:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in batch.columns:
                        batch[dummy_col] = 0

        
            # print(f"[DEBUG] 📦 Ready to apply label. Checking for 'time', 'src_user', etc.")
            # print(f"[DEBUG] Columns now: {list(batch.columns)}")
            
            # Add label
            batch["label"] = batch.apply(
                lambda row: -1 if (row["time"], row["src_user"], row["src_comp"], row["dst_comp"]) in redteam_events else 1,
                axis=1
            )
        
            all_batches.append(batch)
        except Exception as e:
            import traceback
            print(f"[Batch {start}-{start + batch_size}] ❌ Exception caught:\n{traceback.format_exc()}")




    if not all_batches:
        print(f"[Chunk {chunk_id}] ⚠️ No batches processed successfully.")
        return

    chunk = pd.concat(all_batches, ignore_index=True)
    chunk.fillna(0, inplace=True)
    chunk.to_csv(output_file, index=False)
    print(f"[Chunk {chunk_id}] ✅ Saved to {output_file}")


def preprocess_labeled_data_chunked(redteam_file, chunk_dir='data/shared_chunks'):
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    print("[Preprocess] 📥 Loading redteam data...")
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    with gzip.open(redteam_file, 'rt') as redteam:
        redteam_data = pd.read_csv(redteam, names=redteam_columns, sep=',')
    redteam_events = set(zip(redteam_data['time'], redteam_data['user'],
                             redteam_data['src_comp'], redteam_data['dst_comp']))

    output_dir = 'data/labeled_data/chunks'
    os.makedirs(output_dir, exist_ok=True)

    user_freq_path = "data/user_freq.json"
    comp_freq_path = "data/comp_freq.json"

    # 🔒 Check if maps already exist
    if os.path.exists(user_freq_path) and os.path.exists(comp_freq_path):
        print("[Preprocess] ✅ Using cached frequency maps.")
        with open(user_freq_path, "r") as f:
            user_counter = json.load(f)
        with open(comp_freq_path, "r") as f:
            comp_counter = json.load(f)
    else:
        # 🧮 Build global frequency maps
        print("[Preprocess] 🧮 Building global frequency maps for users and computers...")
        user_counter = Counter()
        comp_counter = Counter()
        chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")])
        for chunk_path in chunk_paths:
            df = pd.read_csv(chunk_path, usecols=["src_user", "dst_user", "src_comp", "dst_comp"])
            user_counter.update(df["src_user"].dropna())
            user_counter.update(df["dst_user"].dropna())
            comp_counter.update(df["src_comp"].dropna())
            comp_counter.update(df["dst_comp"].dropna())

        with open(user_freq_path, "w") as f:
            json.dump(user_counter, f)
        with open(comp_freq_path, "w") as f:
            json.dump(comp_counter, f)
        print(f"[Cache] 💾 Saved frequency maps to disk.")

    # 🔁 Process chunks
    chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")])
    for i, chunk_path in enumerate(chunk_paths):
        process_labeled_chunk(
            chunk_path,
            redteam_events,
            categorical_columns,
            output_dir,
            chunk_id=i,
            user_counter=user_counter,
            comp_counter=comp_counter
        )

    print(f"[Preprocess] ✅ Done. {len(chunk_paths)} labeled chunks saved to: {output_dir}")


if __name__ == "__main__":
    print("[Preprocess] 🚀 Starting labeled preprocessing using shared chunks...")
    redteam_file_path = 'data/redteam.txt.gz'
    preprocess_labeled_data_chunked(redteam_file_path)
    print("[Preprocess] 🏁 Labeled preprocessing completed.")
