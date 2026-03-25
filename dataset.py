import json
import warnings
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


PAD_TOKEN = "<pad>"  # Reserved id 0 for padding.
UNK_TOKEN = "<unk>"  # Reserved id 1 for unknown characters.


def _build_vocab(words: Iterable[str]) -> Dict[str, int]:
    # Build a character-level vocab from all words.
    chars = {ch for w in words if isinstance(w, str) for ch in w}
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch in sorted(chars):
        vocab.setdefault(ch, len(vocab))
    return vocab


class SourGrapeDataset(Dataset):
    def __init__(
        self,
        condition: str,
        data_path: str = "dataset/jitter_meta_file.csv",
        expected_word_len: int = 5,
        expected_trajectory_len: int = 122,
    ) -> None:
        # Read metadata and filter to a single condition.
        df = pd.read_csv(data_path)
        df = df[df["condition"] == condition]
        if df.empty:
            raise ValueError(f"No rows found for condition '{condition}'.")
        
        # Store item types for lookup.
        self.item_types = df["item_type"].tolist()
        
        # Look for trajectory paths
        base_dir = Path(data_path).resolve().parent  # Resolve relative trajectory paths.
        sequences = []
        for rel_path in df["jitter_filename"].tolist():
            npy_path = base_dir / str(rel_path)
            arr = np.load(str(npy_path))
            flat = np.asarray(arr).reshape(-1)
            if flat.shape[0] != expected_trajectory_len:
                # Warn and fail fast if the trajectory length is incorrect.
                warnings.warn(
                    f"Expected {expected_trajectory_len} values in {npy_path}, "
                    f"got {flat.shape[0]}.",
                    stacklevel=2,
                )
                raise ValueError(
                    f"All trajectories must be length {expected_trajectory_len}."
                )
            sequences.append(flat.astype(np.float32))
        traj_matrix = np.vstack(sequences)
        self.trajectory_len = expected_trajectory_len

        # Build character vocab and encode each word to ids.
        words = df["UR"].tolist()
        self.vocab = _build_vocab(words)
        # Inverse vocab for decoding ids back to characters.
        self.id_to_char = {idx: ch for ch, idx in self.vocab.items()}
        unk_id = self.vocab[UNK_TOKEN]
        encoded = [
            [self.vocab.get(ch, unk_id) for ch in (w if isinstance(w, str) else "")]
            for w in words
        ]
        bad = [w for w in words if not isinstance(w, str) or len(w) != expected_word_len]
        if bad:
            # Warn and fail fast if any word length violates the requirement.
            warnings.warn(
                f"Found {len(bad)} words not length {expected_word_len} in 'UR'.",
                stacklevel=2,
            )
            raise ValueError(f"All 'UR' words must be length {expected_word_len}.")
        self.word_len = expected_word_len

        # Final tensors ready for DataLoader batching (assumes fixed word length).
        self.x = torch.tensor(encoded, dtype=torch.long)
        # Store real targets and initialize previous-generation targets to real ones.
        self.y_real = torch.tensor(traj_matrix, dtype=torch.float32)
        self.y_prev = self.y_real.clone()

    def __len__(self) -> int:
        # Number of samples.
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Single training example.
        return {
            "x": self.x[idx],
            "y": self.y_prev[idx],
            "y_real": self.y_real[idx],
            "y_prev": self.y_prev[idx],
            "item_type": self.item_types[idx],
        }

    def save_vocab(self, path: str) -> None:
        # Save vocab for reproducible inference.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=True, indent=2)

    def update_prev_targets(self, y_prev: torch.Tensor) -> None:
        # Update previous-generation targets using a full prediction matrix.
        if y_prev.shape != self.y_real.shape:
            raise ValueError("y_prev must have the same shape as y_real.")
        self.y_prev = y_prev.detach().cpu()

    def split_dataset(self, seed: int = 42) -> tuple[Dataset, Dataset, Dataset]:
        # Split this dataset into three roughly even portions.
        return random_split(
            self,
            [1 / 3, 1 / 3, 1 / 3],
            generator=torch.Generator().manual_seed(seed),
        )
