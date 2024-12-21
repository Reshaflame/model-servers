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

# Column definitions for LANL dataset
redteam_columns = ['time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 'action']

# File paths
file_paths = {
    'redteam': 'data/redteam.txt.gz',
    'auth': 'data/sampled_data/auth_sample.csv',
    'proc': 'data/sampled_data/proc_sample.csv',
    'flows': 'data/sampled_data/flows_sample.csv',
    'dns': 'data/sampled_data/dns_sample.csv'
}

# Load redteam data for labels
redteam_data = read_gz_file(file_paths['redteam'], redteam_columns, sample_fraction=1.0)

# Load sampled data
auth_data = pd.read_csv(file_paths['auth'])
proc_data = pd.read_csv(file_paths['proc'])
flows_data = pd.read_csv(file_paths['flows'])
dns_data = pd.read_csv(file_paths['dns'])

# Merge datasets on time or other relevant keys for labeling
# Assuming 'time' is the primary key to merge on
auth_labeled = auth_data.merge(redteam_data, on=['time'], how='left', indicator=True)
auth_labeled['label'] = (auth_labeled['_merge'] == 'both').astype(int)
auth_labeled = auth_labeled.drop(columns=['_merge'])

# Repeat similar merging for other datasets
proc_labeled = proc_data.merge(redteam_data, on=['time'], how='left', indicator=True)
proc_labeled['label'] = (proc_labeled['_merge'] == 'both').astype(int)
proc_labeled = proc_labeled.drop(columns=['_merge'])

flows_labeled = flows_data.merge(redteam_data, on=['time'], how='left', indicator=True)
flows_labeled['label'] = (flows_labeled['_merge'] == 'both').astype(int)
flows_labeled = flows_labeled.drop(columns=['_merge'])

dns_labeled = dns_data.merge(redteam_data, on=['time'], how='left', indicator=True)
dns_labeled['label'] = (dns_labeled['_merge'] == 'both').astype(int)
dns_labeled = dns_labeled.drop(columns=['_merge'])

# Save labeled data
output_dir = 'data/labeled_data'
os.makedirs(output_dir, exist_ok=True)

auth_labeled.to_csv(f'{output_dir}/auth_labeled.csv', index=False)
proc_labeled.to_csv(f'{output_dir}/proc_labeled.csv', index=False)
flows_labeled.to_csv(f'{output_dir}/flows_labeled.csv', index=False)
dns_labeled.to_csv(f'{output_dir}/dns_labeled.csv', index=False)

print("Labeled data saved!")
