import os
import pandas as pd
import gzip

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

def process_labeled_chunk(chunk_path, redteam_events, categorical_columns, output_csv, first_chunk):
    df = pd.read_csv(chunk_path)

    if CUDF_AVAILABLE:
        chunk = cudf.from_pandas(df)
        chunk = chunk.dropna(subset=["time"])
        chunk = cudf.get_dummies(chunk, columns=categorical_columns)
        chunk = chunk.to_pandas()
    else:
        chunk = df.dropna(subset=["time"])
        chunk = pd.get_dummies(chunk, columns=categorical_columns)

    chunk['label'] = chunk.apply(
        lambda row: -1 if (row['time'], row['src_user'], row['src_comp'], row['dst_comp']) in redteam_events else 1,
        axis=1
    )
    chunk.fillna(0, inplace=True)

    chunk.to_csv(output_csv, mode='a', header=first_chunk, index=False)
    print(f"✅ Labeled and appended {chunk_path}")

def preprocess_labeled_data_chunked(redteam_file, chunk_dir='data/shared_chunks'):
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    # Load redteam file once
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    print("[Preprocess] Loading redteam data...")
    with gzip.open(redteam_file, 'rt') as redteam:
        redteam_data = pd.read_csv(redteam, names=redteam_columns, sep=',')
    redteam_events = set(zip(redteam_data['time'], redteam_data['user'], redteam_data['src_comp'], redteam_data['dst_comp']))

    # Preprocess chunks
    output_dir = 'data/labeled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'labeled_auth.csv')
    if os.path.exists(output_csv):
        os.remove(output_csv)

    chunk_paths = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")])
    for i, chunk_path in enumerate(chunk_paths):
        process_labeled_chunk(chunk_path, redteam_events, categorical_columns, output_csv, first_chunk=(i == 0))

    print(f"✅ Labeled preprocessing completed. Final CSV at {output_csv}")

if __name__ == "__main__":
    print("[Preprocess] Starting labeled preprocessing using shared chunks...")
    redteam_file_path = 'data/redteam.txt.gz'
    preprocess_labeled_data_chunked(redteam_file_path)
    print("[Preprocess] Labeled preprocessing completed.")
