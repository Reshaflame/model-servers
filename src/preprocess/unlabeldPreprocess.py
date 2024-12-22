import pandas as pd
import gzip
import os

# Function to extract and read gzip files in chunks
def read_gz_file(file_path, column_names, chunk_size=10**6, sample_fraction=0.01):
    extracted_data = []
    with gzip.open(file_path, 'rt') as file:
        reader = pd.read_csv(file, names=column_names, chunksize=chunk_size, sep=' ')
        for chunk in reader:
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
            extracted_data.append(sampled_chunk)
    return pd.concat(extracted_data, ignore_index=True)

# Function to preprocess time columns
def preprocess_time_column(data, time_column):
    print(f"Raw {time_column} sample:", data[time_column].head())  # Debug raw values
    
    # Extract the first part of the 'time' column
    data[time_column] = data[time_column].str.split(',').str[0]
    print(f"Extracted {time_column} sample:", data[time_column].head())  # Debug extracted values
    
    # Convert to datetime (assuming Unix time)
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce', unit='s')
    print(f"Converted {time_column} sample:", data[time_column].head())  # Debug converted values
    
    # Drop invalid times
    data = data.dropna(subset=[time_column])
    
    # Aggregate time by minute
    data[time_column] = data[time_column].dt.floor('min')
    print(f"Processed {time_column} sample:", data[time_column].head())  # Debug processed values
    
    return data

# Column definitions based on the LANL dataset
auth_columns = ['time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 'auth_type', 'logon_type', 'auth_orientation', 'success']
flows_columns = ['time', 'duration', 'src_comp', 'dst_comp', 'src_port', 'dst_port', 'protocol', 'packets', 'bytes']

# File paths (update with your actual paths)
file_paths = {
    'auth': 'data/auth.txt.gz',
    'flows': 'data/flows.txt.gz'
}

# Sample and load manageable chunks into memory
auth_data = read_gz_file(file_paths['auth'], auth_columns)
flows_data = read_gz_file(file_paths['flows'], flows_columns)

# Preprocess time columns
auth_data = preprocess_time_column(auth_data, 'time')
flows_data = preprocess_time_column(flows_data, 'time')

# Retain only necessary columns
auth_data = auth_data[['time', 'auth_type', 'logon_type', 'auth_orientation', 'success']]
flows_data = flows_data[['time', 'duration', 'packets', 'bytes']]

# Print sample sizes for verification
print("Auth Data Sample:", auth_data.shape)
print("Flows Data Sample:", flows_data.shape)

# Save sampled data as intermediate files for faster access
output_dir = 'data/sampled_data'
os.makedirs(output_dir, exist_ok=True)

auth_data.to_csv(f'{output_dir}/auth_sample.csv', index=False)
flows_data.to_csv(f'{output_dir}/flows_sample.csv', index=False)
print("Sampled data saved!")
