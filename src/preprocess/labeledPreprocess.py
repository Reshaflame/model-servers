import os
import gzip

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    CUDF_AVAILABLE = False

from concurrent.futures import ProcessPoolExecutor

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def process_chunk(chunk, redteam_events, categorical_columns):
    """
    Process a single chunk: label and encode.
    """
    if CUDF_AVAILABLE:
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

def preprocess_labeled_data_with_matching_parallel(auth_file, redteam_file, chunk_size=10**6, output_file='labeled_auth.csv'):
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']

    output_dir = 'data/labeled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    print("Loading redteam data...")
    with gzip.open(redteam_file, 'rt') as redteam:
        redteam_data = pd.read_csv(redteam, names=redteam_columns, sep=',')
    redteam_events = set(zip(redteam_data['time'], redteam_data['user'], redteam_data['src_comp'], redteam_data['dst_comp']))

    total_data = []

    print("Processing authentication data with parallel sampling and matching...")
    with gzip.open(auth_file, 'rt') as auth:
        auth_reader = pd.read_csv(auth, names=columns, sep=',', chunksize=chunk_size)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, chunk, redteam_events, categorical_columns) for chunk in auth_reader]
            for future in futures:
                processed_chunk = future.result()
                total_data.append(processed_chunk)

    if total_data:
        final_data = pd.concat(total_data, ignore_index=True)
        print(f"Combined dataset shape: {final_data.shape}")
        print(f"[Preprocess] Anomaly samples: {final_data[final_data['label'] == -1].shape[0]}")
        print(f"[Preprocess] Normal samples: {final_data[final_data['label'] == 1].shape[0]}")

        print(f"Saving labeled dataset to {output_path}...")
        final_data.to_csv(output_path, index=False)
        print("Labeled dataset saved.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    print("Starting labeled preprocessing with fallback mechanism...")
    auth_file_path = 'data/auth.txt.gz'
    redteam_file_path = 'data/redteam.txt.gz'
    preprocess_labeled_data_with_matching_parallel(auth_file_path, redteam_file_path, chunk_size=10**6)
    print("Labeled preprocessing completed.")
