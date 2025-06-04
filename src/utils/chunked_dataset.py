# checked
import os
import pandas as pd
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from glob import glob

class ChunkedCSVDataset(IterableDataset):
    def __init__(self, chunk_dir, chunk_size=10000, label_col='label', device='cpu', expected_features=None):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.label_col = label_col
        self.device = device
        self.expected_features = expected_features

        # Safely collect only valid files
        self.chunk_files = sorted(glob(os.path.join(chunk_dir, "*.csv")))
        if not self.chunk_files:
            raise FileNotFoundError(f"No CSV chunks found in {chunk_dir}")

        print(f"[Dataset] ✅ {len(self.chunk_files)} chunk files found.")

    def _batch_generator(self, chunk_df):
        total_rows = len(chunk_df)
        for start in range(0, total_rows, self.chunk_size):
            batch = chunk_df.iloc[start:start + self.chunk_size]
            labels = (batch[self.label_col] == -1).astype(float)
            features = batch.drop(columns=[self.label_col])
            yield features.to_numpy(), labels.to_numpy()

    
    def _infer_input_size(self):
        # Use the first file to infer feature dimensions
        sample = pd.read_csv(self.chunk_files[0])
        sample = sample.drop(columns=[self.label_col], errors='ignore')
        return sample.shape[1]

    def __iter__(self):
        for chunk_path in self.chunk_files:
            try:
                chunk = pd.read_csv(chunk_path)
    
                if chunk.empty:
                    print(f"[⚠️] Skipping empty chunk: {chunk_path}")
                    continue
    
                if self.label_col not in chunk.columns:
                    print(f"[⛔] 'label' column missing in chunk: {chunk_path}")
                    continue
    
                if self.expected_features:
                    missing = [col for col in self.expected_features if col not in chunk.columns]
                    for col in missing:
                        chunk[col] = 0
                    chunk = chunk[self.expected_features + [self.label_col]]
    
                yield from self._batch_generator(chunk)
    
            except Exception as e:
                print(f"[❌] Error loading chunk {chunk_path}: {e}")


    def train_loader(self):
        return DataLoader(self, batch_size=None)

    def val_loader(self):
        return DataLoader(self, batch_size=None)

    def full_loader(self):
        return DataLoader(self, batch_size=None)
    
    @property
    def input_size(self):
        # Load just one chunk to infer input size
        sample_chunk = pd.read_csv(self.chunk_files[0], nrows=1)
        return sample_chunk.drop(columns=[self.label_col]).shape[1]
