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

        # keep only chunks that contain at least one anomaly
        positive_chunks = [p for p in self.chunk_paths if self._chunk_has_pos(p)]
        dropped         = len(self.chunk_paths) - len(positive_chunks)
        if dropped:
            print(f"[Init] ⚖️  Skipping {dropped} chunks with 0 anomalies")
        if not positive_chunks:
            raise RuntimeError("No chunks with anomalies found!")
        self.chunk_paths = positive_chunks

        # Detect input size from first chunk
        sample_df = pd.read_csv(self.chunk_paths[0])
        numeric_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col != self.label_column]
        print(f"[Init] 🧬 Using {len(self.feature_columns)} numeric features: {self.feature_columns[:5]}{'...' if len(self.feature_columns) > 5 else ''}")

        self.input_size = len(self.feature_columns)

    def _chunk_has_pos(self, chunk_path):
        df = pd.read_csv(chunk_path, usecols=[self.label_column])
        return (df[self.label_column] == 1).any()

    def resplit(self):
        from random import shuffle
        shuffle(self.chunk_paths)

    def _load_sequences_from_chunk(self, chunk_path):
        # ------------------------------------------------------
        # 1) Load CSV (python engine avoids rare tokenising errors)
        # ------------------------------------------------------
        df = pd.read_csv(chunk_path, engine="python")

        # Warn if still any non-numerics sneaked in
        if df[self.feature_columns].select_dtypes(include="object").shape[1] > 0:
            print(f"[WARNING] ⚠️  {os.path.basename(chunk_path)} "
                  "contains non-numeric cols – coerced to 0.")


        if self.binary_labels:
            df[self.label_column] = (df[self.label_column] ==  1).astype(float)

        # ------------------------------------------------------
        # 3) Build sequences
        # ------------------------------------------------------
        feats = (df[self.feature_columns]
                 .apply(pd.to_numeric, errors="coerce")
                 .fillna(0.0)
                 .values)
        labels = df[self.label_column].values

        X_seq, y_seq = [], []
        for i in range(len(feats) - self.sequence_length):
            X_seq.append(feats[i:i + self.sequence_length])
            y_seq.append(labels[i + self.sequence_length - 1])

        X_tensor = torch.tensor(np.asarray(X_seq,  dtype=np.float32),
                                device=self.device)
        y_tensor = torch.tensor(np.asarray(y_seq,  dtype=np.float32),
                                device=self.device).unsqueeze(1)  # (N,1)

        dataset = TensorDataset(X_tensor, y_tensor)

        # ------------------------------------------------------
        # 4) Train / Val split  (random_split stores *indices*)
        # ------------------------------------------------------
        train_len = int(len(dataset) * self.split_ratio)
        val_len   = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        # ----  Guarantee ≥1 anomaly in validation split  -----
        if self.binary_labels:
            y_train = y_tensor[train_set.indices]
            y_val   = y_tensor[val_set.indices]

            if (y_val == 1).sum() == 0 and (y_train == 1).sum() > 0:
                # absolute index of the first positive inside train_set
                pos_abs_idx = train_set.indices[(y_train == 1)
                                                .nonzero(as_tuple=False)[0, 0]
                                                .item()]
                train_set.indices.remove(pos_abs_idx)
                val_set.indices.append(pos_abs_idx)

        # ------------------------------------------------------
        # 5) WeightedRandomSampler on *train* loader
        # ------------------------------------------------------
        sampler = None
        if self.binary_labels:
            lbl_train = y_tensor[train_set.indices]
            num_pos   = int((lbl_train == 1).sum().item())
            num_neg   = int((lbl_train == 0).sum().item())

            if num_pos == 0:
                print(f"[Warning] No anomalies in {os.path.basename(chunk_path)}")
            else:
                # weight  =  (#neg / #pos)  for positive samples,  1.0 for normal
                weights = (lbl_train == 1).float() * (num_neg / (num_pos + 1e-6)) + 1.0
                weights = weights.squeeze()          # (N,)  → required by sampler
                sampler = WeightedRandomSampler(weights,
                                                num_samples=len(weights),
                                                replacement=True)

        # ------------------------------------------------------
        # 6) DataLoaders
        # ------------------------------------------------------
        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=sampler is None)

        val_loader   = DataLoader(val_set,
                                  batch_size=self.batch_size,
                                  shuffle=False)

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

