import os
import gzip
import torch

CUDF_AVAILABLE = False

if torch.cuda.is_available():
    try:
        import cudf
        CUDF_AVAILABLE = True
        print("[INFO] cuDF detected and CUDA available.")
    except ImportError:
        import pandas as pd
        print("[INFO] cuDF not found, falling back to pandas.")
else:
    import pandas as pd
    print("[INFO] CUDA not available, using pandas.")


def preprocess_auth_data(file_path, chunk_size=10**6, output_file='auth.csv'):
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    output_dir = 'data/sampled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    categorical_columns = ['auth_type', 'logon_type', 'auth_orientation', 'success']
    all_categories = {col: set() for col in categorical_columns}

    print("Determining unique categories...")
    with gzip.open(file_path, 'rt') as file:
        for chunk in pd.read_csv(file, names=columns, sep=',', chunksize=chunk_size):
            for col in categorical_columns:
                all_categories[col].update(chunk[col].dropna().unique())

    print("Unique categories found:")
    for col, categories in all_categories.items():
        print(f"{col}: {categories}")

    with gzip.open(file_path, 'rt') as file:
        chunk_iterator = pd.read_csv(file, names=columns, sep=',', chunksize=chunk_size)
        for i, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {i+1}...")

            if CUDF_AVAILABLE:
                chunk = cudf.from_pandas(chunk)
                chunk['time'] = cudf.to_datetime(chunk['time'], errors='coerce', unit='s')
                chunk = chunk.dropna(subset=['time'])

                for col in categorical_columns:
                    chunk[col] = chunk[col].astype('category').cat.set_categories(list(all_categories[col]))
                chunk = cudf.get_dummies(chunk, columns=categorical_columns, dummy_na=False)
                chunk = chunk.fillna(0)
                chunk = chunk.to_pandas()

            else:
                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce', unit='s')
                chunk = chunk.dropna(subset=['time'])

                for col in categorical_columns:
                    chunk[col] = pd.Categorical(chunk[col], categories=all_categories[col])
                chunk = pd.get_dummies(chunk, columns=categorical_columns, dummy_na=False)
                chunk = chunk.fillna(0)

            # Stream to CSV to avoid high memory usage
            if not os.path.exists(output_path):
                chunk.to_csv(output_path, index=False)
            else:
                chunk.to_csv(output_path, mode='a', header=False, index=False)

    print(f"Preprocessing completed and saved to {output_path}")

if __name__ == "__main__":
    print("Starting preprocessing...")
    auth_file_path = 'data/auth.txt.gz'
    preprocess_auth_data(auth_file_path)
    print("Preprocessing completed.")
