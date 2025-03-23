import os
import gzip

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    CUDF_AVAILABLE = False


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

    sampled_data = []
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
                sampled_data.append(chunk.to_pandas())

            else:
                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce', unit='s')
                chunk = chunk.dropna(subset=['time'])

                for col in categorical_columns:
                    chunk[col] = pd.Categorical(chunk[col], categories=all_categories[col])
                chunk = pd.get_dummies(chunk, columns=categorical_columns, dummy_na=False)
                chunk = chunk.fillna(0)
                sampled_data.append(chunk)

    if sampled_data:
        final_data = pd.concat(sampled_data, ignore_index=True)
        print(f"Combined dataset shape: {final_data.shape}")
        print("Missing values after combining chunks:", final_data.isna().sum())
        print(f"Overwriting existing file: {output_path}")
        final_data.to_csv(output_path, index=False)
        print(f"Processed sample saved to {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    print("Starting preprocessing...")
    auth_file_path = 'data/auth.txt.gz'
    preprocess_auth_data(auth_file_path)
    print("Preprocessing completed.")
