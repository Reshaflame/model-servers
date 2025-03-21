import pandas as pd
import gzip
import os

def preprocess_auth_data_sample(file_path, sample_fraction=0.01, chunk_size=10**6, output_file='auth_sample.csv'):
    # Define column names
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    output_dir = 'data/sampled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Determine all possible categories for each categorical column
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

    # Start processing chunks with consistent columns
    sampled_data = []
    with gzip.open(file_path, 'rt') as file:
        chunk_iterator = pd.read_csv(file, names=columns, sep=',', chunksize=chunk_size)
        for i, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {i+1}...")

            # Take a random sample
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)

            # Preprocess sampled chunk
            sampled_chunk['time'] = pd.to_datetime(sampled_chunk['time'], errors='coerce', unit='s')
            sampled_chunk = sampled_chunk.dropna(subset=['time'])

            # One-hot encode categorical columns with predefined categories
            for col in categorical_columns:
                sampled_chunk[col] = pd.Categorical(sampled_chunk[col], categories=all_categories[col])
            sampled_chunk = pd.get_dummies(sampled_chunk, columns=categorical_columns, dummy_na=False)

            # Fill NaN values in all columns with 0
            sampled_chunk.fillna(0, inplace=True)

            sampled_data.append(sampled_chunk)

    # Combine all sampled chunks into a single DataFrame
    if sampled_data:
        final_data = pd.concat(sampled_data, ignore_index=True)
        print(f"Combined dataset shape: {final_data.shape}")
        print("Missing values after combining chunks:", final_data.isna().sum())

        # Overwrite the existing file without merging
        print(f"Overwriting existing file: {output_path}")
        final_data.to_csv(output_path, index=False)
        print(f"Processed 1% sample saved to {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    print("Starting preprocessing...")
    auth_file_path = 'data/auth.txt.gz'
    preprocess_auth_data_sample(auth_file_path, sample_fraction=0.01)
    print("Preprocessing completed.")
