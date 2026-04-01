import json
import warnings
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from hyper_params import HyperParams
from preprocessing import augment_trajectory_variable_length


PAD_TOKEN = "<pad>"  # Reserved id 0 for padding.
UNK_TOKEN = "<unk>"  # Reserved id 1 for unknown characters.


class SourGrapeDataset(Dataset):
    def __init__(
        self,
        condition: str,
        data_path: str,
        expected_word_len: int = HyperParams().expected_word_len,
        trajectory_pad_value: float = HyperParams().trajectory_pad_value,
        max_trajectory_len: int = HyperParams().max_trajectory_len,
        augment: bool = False,
    ) -> None:
        # Read metadata and filter to a single condition.
        df = pd.read_csv(data_path)
        df = df[df["condition"] == condition]
        
        # Store item types for lookup.
        self.item_types = df["item_type"].tolist()
        
        # Load trajectory paths.
        base_dir = Path(data_path).resolve().parent  # Resolve relative trajectory paths.
        sequences = []
        for rel_path in df["file_name"].tolist():
            npy_path = base_dir / str(rel_path)
            arr = np.load(str(npy_path))
            flat = np.asarray(arr).reshape(-1)
            if augment:
                aug = augment_trajectory_variable_length(
                    torch.tensor(flat).unsqueeze(1)
                )
                flat = aug.squeeze(1).detach().cpu().numpy()
            if flat.shape[0] > max_trajectory_len:
                raise ValueError(
                    f"Trajectory length {flat.shape[0]} exceeds max {max_trajectory_len}."
                )
            sequences.append(flat.astype(np.float32))
        max_len = max(len(s) for s in sequences) if sequences else 0
        padded = [
            self._pad_trajectory(s, max_len, trajectory_pad_value) for s in sequences
        ]
        traj_matrix = np.vstack(padded) if padded else np.zeros((0, max_len), dtype=np.float32)
        self.trajectory_len = max_len

        # Build character vocab and encode each word to ids.
        words = df["word"].tolist()
        self.words = words
        self.vocab, self.id_to_char = self._build_vocab(words)
        unk_id = self.vocab[UNK_TOKEN]
        encoded = [
            [self.vocab.get(ch, unk_id) for ch in (w if isinstance(w, str) else "")]
            for w in words
        ]
        bad = [w for w in words if not isinstance(w, str) or len(w) != expected_word_len]
        if bad:
            warnings.warn(
                f"Found {len(bad)} words not length {expected_word_len} in 'word'.",
                stacklevel=2,
            )
            raise ValueError(f"All 'word' values must be length {expected_word_len}.")
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
            "word": self.words[idx],
        }

    def _pad_trajectory(
        self,
        flat: np.ndarray,
        pad_to_len: int,
        trajectory_pad_value: float,
    ) -> np.ndarray:
        # Pad a single trajectory to the target length (no truncation).
        if flat.shape[0] < pad_to_len:
            pad = np.full(
                pad_to_len - flat.shape[0],
                trajectory_pad_value,
                dtype=flat.dtype,
            )
            flat = np.concatenate([flat, pad], axis=0)
        return flat

    def _build_vocab(self, words: Iterable[str]) -> tuple[Dict[str, int], Dict[int, str]]:
        # Build a character-level vocab from all words and its inverse map.
        chars = {ch for w in words if isinstance(w, str) for ch in w}
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for ch in sorted(chars):
            vocab.setdefault(ch, len(vocab))
        id_to_char = {idx: ch for ch, idx in vocab.items()}
        return vocab, id_to_char

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
