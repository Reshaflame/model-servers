import gzip
import os

input_path = "/home/reshaflame/model-servers/data/auth.txt.gz"
output_prefix = "auth_quarter"
output_dir = "./"
num_parts = 4

# Count total lines (excluding header)
with gzip.open(input_path, 'rt') as f:
    total_lines = sum(1 for _ in f)

lines_per_chunk = total_lines // num_parts

print(f"[Split] Total lines: {total_lines}, ~{lines_per_chunk} per chunk")

with gzip.open(input_path, 'rt') as infile:
    for part in range(num_parts):
        output_path = os.path.join(output_dir, f"{output_prefix}_{part+1:02d}.txt.gz")
        with gzip.open(output_path, 'wt') as out:
            for i in range(lines_per_chunk):
                line = infile.readline()
                if not line:
                    break
                out.write(line)
        print(f"✅ Saved: {output_path}")
