import os
import gzip
import json
import pandas as pd
from pathlib import Path

class DataChunker:
    def __init__(self, input_file, output_dir, chunk_size=500_000, category_path="data/auth_categories.json"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.columns = [
            "time", "src_user", "dst_user", "src_comp", "dst_comp",
            "auth_type", "logon_type", "auth_orientation", "success"
        ]
        with open(category_path) as f:
            self.auth_categories = json.load(f)

    def enforce_categories(self, chunk):
        """
        Sets category values explicitly to ensure consistent dummy columns later.
        """
        for cat_col in self.auth_categories:
            if cat_col in chunk.columns:
                chunk[cat_col] = pd.Categorical(
                    chunk[cat_col].fillna("?"),
                    categories=self.auth_categories[cat_col]
                )
        return chunk

    def chunk_and_save(self):
        """
        Splits a large dataset into smaller CSV chunks stored on disk,
        using fixed categories to ensure consistency across all chunks.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[DataChunker] 🔧 Splitting {self.input_file} into chunks with enforced categories...")

        with gzip.open(self.input_file, 'rt') as infile:
            chunk_iter = pd.read_csv(
                infile,
                names=self.columns,
                sep=",",
                na_values=["?"],  # Treat "?" as NaN
                chunksize=self.chunk_size,
                dtype=str,  # Avoid dtype inference issues
                on_bad_lines='skip'
            )

            for idx, chunk in enumerate(chunk_iter):
                chunk = self.enforce_categories(chunk)
                chunk_path = os.path.join(self.output_dir, f"chunk_{idx:03}.csv")
                chunk.to_csv(chunk_path, index=False)
                print(f"[DataChunker] 💾 Saved {chunk.shape[0]} rows to {chunk_path}")

        print(f"[DataChunker] ✅ Done splitting into chunks in {self.output_dir}")

if __name__ == "__main__":
    chunker = DataChunker(
        input_file="data/auth.txt.gz",
        output_dir="data/shared_chunks",
        chunk_size=500_000,
        category_path="data/auth_categories.json"
    )
    chunker.chunk_and_save()
