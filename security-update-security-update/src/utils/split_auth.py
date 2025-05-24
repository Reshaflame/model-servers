# src/utils/split_auth.py

import os
import pandas as pd

def chunk_auth_file(
    input_path='data/auth.txt.gz',
    output_dir='data/shared_chunks',
    rows_per_chunk=1_000_000
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Chunker] 📂 Reading from {input_path}")

    # Define correct column names
    columns = [
        "time", "src_user", "dst_user", "src_comp", "dst_comp",
        "auth_type", "logon_type", "auth_orientation", "success"
    ]

    chunk_iter = pd.read_csv(
        input_path,
        sep=",",                     # 🛠️ Critical fix
        names=columns,              # 🛠️ Add headers manually
        chunksize=rows_per_chunk,
        dtype=str,
        compression='gzip'
    )

    for i, chunk in enumerate(chunk_iter):
        chunk_file = os.path.join(output_dir, f"chunk_{i:03}.csv")
        chunk.to_csv(chunk_file, index=False)
        print(f"[Chunker] ✅ Saved chunk {i+1} to {chunk_file}")

    print(f"[Chunker] 🎉 All done! Chunks saved to {output_dir}")

if __name__ == "__main__":
    chunk_auth_file()
