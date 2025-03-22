import pandas as pd
import gzip
import os
from concurrent.futures import ProcessPoolExecutor

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def encode_categorical_columns(data, categorical_columns):
    """
    Encode categorical columns by turning categories into numerical IDs or one-hot encoding.
    """
    for col in categorical_columns:
        data[col] = pd.Categorical(data[col]).codes  # Assign numerical IDs
    return data

def process_chunk(chunk, redteam_events):
    """
    Process a single chunk: sample, label, and encode.
    """
    # Sampling for DEV
    # sampled_chunk = chunk.sample(frac=0.01, random_state=42)

    # Sampling for Runpod
    sampled_chunk = chunk

    # Labeling
    sampled_chunk['label'] = sampled_chunk.apply(
        lambda row: -1 if (row['time'], row['src_user'], row['src_comp'], row['dst_comp']) in redteam_events else 1,
        axis=1
    )

    # Encode categorical columns into numerical values
    categorical_columns = ['src_user', 'dst_user', 'src_comp', 'dst_comp', 'auth_type', 'logon_type', 'auth_orientation', 'success']
    sampled_chunk = encode_categorical_columns(sampled_chunk, categorical_columns)

    # Fill NaN values
    sampled_chunk.fillna(0, inplace=True)

    return sampled_chunk

def preprocess_labeled_data_with_matching_parallel(auth_file, redteam_file, chunk_size=10**6, output_file='labeled_auth_sample.csv'):
    # Define column names
    auth_columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    output_dir = 'data/labeled_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Load redteam data
    print("Loading redteam data...")
    with gzip.open(redteam_file, 'rt') as redteam:
        redteam_data = pd.read_csv(redteam, names=redteam_columns, sep=',')
    redteam_events = set(zip(redteam_data['time'], redteam_data['user'], redteam_data['src_comp'], redteam_data['dst_comp']))

    total_data = []
    matches_found = 0

    print("Processing authentication data with parallel sampling and matching...")
    with gzip.open(auth_file, 'rt') as auth:
        auth_reader = pd.read_csv(auth, names=auth_columns, sep=',', chunksize=chunk_size)

        # Process chunks in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, chunk, redteam_events) for chunk in auth_reader]

            for future in futures:
                processed_chunk = future.result()
                total_data.append(processed_chunk)

                # Count matches in this chunk
                chunk_matches = processed_chunk[processed_chunk['label'] == -1].shape[0]
                matches_found += chunk_matches

                # Stop sampling more chunks once sufficient matches are found
                # if matches_found >= 5:
                #     print(f"Sufficient matches found ({matches_found}). Stopping further sampling.")
                #     break

    # Combine all processed data
    if total_data:
        final_data = pd.concat(total_data, ignore_index=True)
        print(f"Combined dataset shape: {final_data.shape}")

        # Save the labeled dataset
        print(f"Saving labeled dataset to {output_path}...")
        final_data.to_csv(output_path, index=False)
        print("Labeled dataset saved.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    # Add project root directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    print("Starting labeled preprocessing with parallel sampling and matching...")
    auth_file_path = 'data/auth.txt.gz'
    redteam_file_path = 'data/redteam.txt.gz'
    preprocess_labeled_data_with_matching_parallel(auth_file_path, redteam_file_path, chunk_size=10**6)
    print("Labeled preprocessing completed.")
