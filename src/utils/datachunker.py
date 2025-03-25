
import os
import gzip
import pandas as pd
from pathlib import Path

class DataChunker:
    def __init__(self, input_file, output_dir, chunk_size=500_000):
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.columns = [
            "time", "src_user", "dst_user", "src_comp", "dst_comp",
            "auth_type", "logon_type", "auth_orientation", "success"
        ]

    def chunk_and_save(self):
        """
        Splits a large dataset into smaller CSV chunks stored on disk.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[DataChunker] Splitting {self.input_file} into chunks...")

        with gzip.open(self.input_file, 'rt') as infile:
            chunk_iter = pd.read_csv(infile, names=self.columns, sep=',', chunksize=self.chunk_size)
            for idx, chunk in enumerate(chunk_iter):
                chunk_path = os.path.join(self.output_dir, f"chunk_{idx}.csv")
                chunk.to_csv(chunk_path, index=False)
                print(f"[DataChunker] Saved {chunk.shape[0]} rows to {chunk_path}")

        print(f"[DataChunker] ✅ Done splitting into chunks in {self.output_dir}")

if __name__ == "__main__":
    # Example usage for manual test
    chunker = DataChunker(
        input_file="data/auth.txt.gz",
        output_dir="data/shared_chunks",
        chunk_size=500_000
    )
    chunker.chunk_and_save()
