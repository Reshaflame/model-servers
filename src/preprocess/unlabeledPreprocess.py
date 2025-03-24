import os
import pandas as pd
import torch
from utils.datachunker import DataChunker

try:
    import cudf
    CUDF_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDF_AVAILABLE = False


def preprocess_chunk(chunk_path, output_csv, all_categories, first_chunk):
    df = pd.read_csv(chunk_path)

    # Ensure time column is handled properly
    if CUDF_AVAILABLE:
        chunk = cudf.from_pandas(df)
        chunk['time'] = cudf.to_datetime(chunk['time'], errors='coerce', unit='s')
        chunk = chunk.dropna(subset=['time'])
        for col in all_categories:
            chunk[col] = chunk[col].astype('category').cat.set_categories(list(all_categories[col]))
        chunk = cudf.get_dummies(chunk, columns=all_categories, dummy_na=False)
        chunk = chunk.fillna(0)
        chunk = chunk.to_pandas()
    else:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', unit='s')
        df = df.dropna(subset=['time'])
        for col in all_categories:
            df[col] = pd.Categorical(df[col], categories=all_categories[col])
        df = pd.get_dummies(df, columns=all_categories, dummy_na=False)
        df = df.fillna(0)
        chunk = df

    # Append to output CSV
    chunk.to_csv(output_csv, mode='a', header=first_chunk, index=False)
    print(f"✅ Processed and appended {chunk_path}")


def preprocess_auth_data(file_path, chunk_size=500000):
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    # Step 1: Split large file into chunks
    chunk_dir = 'data/chunks_unlabeled'
    os.makedirs(chunk_dir, exist_ok=True)
    chunker = DataChunker(file_path, output_dir=chunk_dir, chunk_size=chunk_size)
    chunk_paths = chunker.split_csv()

    # Step 2: Extract unique categories once
    print("[Preprocess] Scanning all chunks for category values...")
    all_categories = {col: set() for col in categorical_columns}

    for chunk_path in chunk_paths:
        chunk_df = pd.read_csv(chunk_path)
        for col in categorical_columns:
            all_categories[col].update(chunk_df[col].dropna().unique())

    print("[Preprocess] Categories detected:")
    for col, cats in all_categories.items():
        print(f"{col}: {cats}")

    # Step 3: Preprocess each chunk and append to single CSV
    output_dir = 'data/preprocessed_unlabeled'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'auth_preprocessed.csv')

    if os.path.exists(output_csv):
        os.remove(output_csv)

    for i, chunk_path in enumerate(chunk_paths):
        preprocess_chunk(chunk_path, output_csv, all_categories, first_chunk=(i == 0))

    print(f"✅ Preprocessing complete. Final CSV saved to {output_csv}")


if __name__ == "__main__":
    print("[Preprocess] Starting unlabeled preprocessing with chunking...")
    auth_file_path = 'data/auth.txt.gz'
    preprocess_auth_data(auth_file_path)
    print("[Preprocess] Unlabeled preprocessing completed.")
