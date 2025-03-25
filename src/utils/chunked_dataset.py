import os
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import random

class ChunkedCSVDataset(IterableDataset):
    def __init__(self, chunk_dir, chunk_size=10000, label_col='label', device='cpu'):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.label_col = label_col
        self.device = device
        self.chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".csv")]
        random.shuffle(self.chunk_files)  # Shuffle at init

    def __iter__(self):
        for file in self.chunk_files:
            for chunk in pd.read_csv(file, chunksize=self.chunk_size):
                labels = (chunk[self.label_col] != -1).astype(float)
                features = chunk.drop(columns=[self.label_col])

                features_tensor = torch.tensor(features.values, dtype=torch.float32).to(self.device)
                labels_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1).to(self.device)

                yield from zip(features_tensor, labels_tensor)
