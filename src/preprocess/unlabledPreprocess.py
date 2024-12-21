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

# Column definitions based on the LANL dataset
auth_columns = ['time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 'auth_type', 'logon_type', 'auth_orientation', 'success']
proc_columns = ['time', 'host', 'name', 'pid', 'ppid', 'start', 'stop']
flows_columns = ['time', 'duration', 'src_comp', 'dst_comp', 'src_port', 'dst_port', 'protocol', 'packets', 'bytes']
dns_columns = ['time', 'src_comp', 'query_name', 'query_class', 'query_type', 'response_code']

# File paths (update with your actual paths)
file_paths = {
    'auth': 'data/auth.txt.gz',
    'proc': 'data/proc.txt.gz',
    'flows': 'data/flows.txt.gz',
    'dns': 'data/dns.txt.gz'
}

# Sample and load manageable chunks into memory
auth_data = read_gz_file(file_paths['auth'], auth_columns)
proc_data = read_gz_file(file_paths['proc'], proc_columns)
flows_data = read_gz_file(file_paths['flows'], flows_columns)
dns_data = read_gz_file(file_paths['dns'], dns_columns)

# Print sample sizes for verification
print("Auth Data Sample:", auth_data.shape)
print("Proc Data Sample:", proc_data.shape)
print("Flows Data Sample:", flows_data.shape)
print("DNS Data Sample:", dns_data.shape)

# Save sampled data as intermediate files for faster access
output_dir = 'data/sampled_data'
os.makedirs(output_dir, exist_ok=True)

auth_data.to_csv(f'{output_dir}/auth_sample.csv', index=False)
proc_data.to_csv(f'{output_dir}/proc_sample.csv', index=False)
flows_data.to_csv(f'{output_dir}/flows_sample.csv', index=False)
dns_data.to_csv(f'{output_dir}/dns_sample.csv', index=False)
print("Sampled data saved!")