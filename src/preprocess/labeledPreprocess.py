import os
import gzip
import pandas as pd
from utils.datachunker import DataChunker

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False


def process_chunk(chunk_path, redteam_events, categorical_columns):
    chunk = pd.read_csv(chunk_path)

    if CUDF_AVAILABLE:
        import cudf
        chunk = cudf.from_pandas(chunk)
        chunk = chunk.dropna(subset=["time"])
        chunk = cudf.get_dummies(chunk, columns=categorical_columns)
        chunk = chunk.to_pandas()
    else:
        chunk = chunk.dropna(subset=["time"])
        chunk = pd.get_dummies(chunk, columns=categorical_columns)

    chunk['label'] = chunk.apply(lambda row: -1 if (row['time'], row['src_user'], row['src_comp'], row['dst_comp']) in redteam_events else 1, axis=1)
    chunk.fillna(0, inplace=True)
    return chunk


def preprocess_labeled_data_chunked(auth_file, redteam_file, output_file='labeled_auth.csv', chunk_size=500_000):
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    output_dir = 'data/labeled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Load redteam data
    print("[Preprocess] Loading redteam data...")
    with gzip.open(redteam_file, 'rt') as redteam:
        redteam_data = pd.read_csv(redteam, names=redteam_columns, sep=',')
    redteam_events = set(zip(redteam_data['time'], redteam_data['user'], redteam_data['src_comp'], redteam_data['dst_comp']))

    # Step 1: Chunk the large dataset
    chunker = DataChunker(input_gz_path=auth_file, output_dir="data/labeled_data/chunks", chunk_size=chunk_size)
    chunk_paths = chunker.split_csv_into_chunks()

    print("[Preprocess] Processing chunks...")
    processed_chunks = []
    for chunk_path in chunk_paths:
        chunk_df = process_chunk(chunk_path, redteam_events, categorical_columns)
        processed_chunks.append(chunk_df)

        # Append chunk directly to final CSV to limit memory usage
        if not os.path.exists(output_path):
            chunk_df.to_csv(output_path, mode='w', index=False)
        else:
            chunk_df.to_csv(output_path, mode='a', header=False, index=False)

    print(f"[Preprocess] ✅ Final preprocessed file saved at: {output_path}")


if __name__ == "__main__":
    print("[Preprocess] Starting labeled preprocessing with chunking...")
    auth_file_path = 'data/auth.txt.gz'
    redteam_file_path = 'data/redteam.txt.gz'
    preprocess_labeled_data_chunked(auth_file_path, redteam_file_path)
    print("[Preprocess] Labeled preprocessing completed.")
