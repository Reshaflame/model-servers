# checked
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler


class SequenceChunkedDataset:
    def __init__(
        self,
        chunk_dir,
        sequence_length=10,
        label_column='label',
        batch_size=128,
        shuffle_files=True,
        binary_labels=True,
        device='cpu',
        split_ratio=0.8,
    ):
        self.chunk_dir = chunk_dir
        self.sequence_length = sequence_length
        self.label_column = label_column
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.binary_labels = binary_labels
        self.device = device
        self.split_ratio = split_ratio

        self.chunk_paths = sorted([
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
            if f.endswith(".csv")
        ])
        if self.shuffle_files:
            from random import shuffle
            shuffle(self.chunk_paths)

        # Detect input size from first chunk
        sample_df = pd.read_csv(self.chunk_paths[0])
        numeric_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col != self.label_column]
        print(f"[Init] 🧬 Using {len(self.feature_columns)} numeric features: {self.feature_columns[:5]}{'...' if len(self.feature_columns) > 5 else ''}")

        self.input_size = len(self.feature_columns)

    def _load_sequences_from_chunk(self, chunk_path):
        df = pd.read_csv(chunk_path)

        if df[self.feature_columns].select_dtypes(include="object").shape[1] > 0:
            print(f"[WARNING] ⚠️ Chunk {os.path.basename(chunk_path)} has non-numeric columns. They will be coerced to 0.")

        if self.binary_labels:
            df[self.label_column] = (df[self.label_column] == -1).astype(float)

        features = df[self.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        labels = pd.to_numeric(df[self.label_column], errors='coerce').fillna(0.0).values

        X_seq, y_seq = [], []
        for i in range(len(features) - self.sequence_length):
            X_seq.append(features[i:i+self.sequence_length])
            y_seq.append(labels[i + self.sequence_length - 1])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)

        # === Train/Val split (still applied before DataLoader) ===
        train_size = int(len(dataset) * self.split_ratio)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        # === Apply WeightedRandomSampler to train loader ===
        if self.binary_labels:
            labels_train = y_tensor[train_set.indices]
            num_pos = (labels_train == 1).sum().item()
            num_neg = (labels_train == 0).sum().item()

            if num_pos == 0:
                print(f"[Warning] No anomalies in chunk {os.path.basename(chunk_path)}")
                sampler = None
            else:
                weights = (labels_train == 1).float() * (num_neg / (num_pos + 1e-6)) + 1.0
                weights = weights.squeeze()  # ✅ FIX: make sure shape is (N,)
                sampler = WeightedRandomSampler(weights, num_samples=len(labels_train), replacement=True)
        else:
            sampler = None

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                sampler=sampler if sampler else None,
                                shuffle=(sampler is None))
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


    def train_loader(self):
        for chunk_path in self.chunk_paths:
            train_loader, _ = self._load_sequences_from_chunk(chunk_path)
            yield from train_loader

    def val_loader(self):
        for chunk_path in self.chunk_paths:
            _, val_loader = self._load_sequences_from_chunk(chunk_path)
            yield from val_loader

    def full_loader(self):
        for chunk_path in self.chunk_paths:
            train_loader, val_loader = self._load_sequences_from_chunk(chunk_path)
            for batch in train_loader:
                yield batch
            for batch in val_loader:
                yield batch

