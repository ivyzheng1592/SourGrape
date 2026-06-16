import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

import hyper_params as hp
from preprocessing import add_noise, augment_trajectory_variable_length


PAD_TOKEN = "<pad>"  # Reserved id 0 for padding.


@dataclass
class Vocab:
    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]

    @classmethod
    def build_vocab(cls, symbols: list[str], pad_id: int) -> "Vocab":
        # Build the vocabulary mappings.
        char_to_id = {PAD_TOKEN: pad_id}
        for symbol in sorted(set(symbols)):
            char_to_id.setdefault(symbol, len(char_to_id))
        id_to_char = {idx: ch for ch, idx in char_to_id.items()}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)

    def save(self, path: str) -> None:
        # Save the vocabulary ids.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=True, indent=2)


class RepeatShuffleSampler(Sampler[int]):
    def __init__(self, dataset_size: int, repeats: int, seed: int) -> None:
        self.dataset_size = dataset_size
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Use a different random seed for each epoch.
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # Repeat each dataset index the requested number of times.
        indices = torch.arange(self.dataset_size, dtype=torch.long).repeat(self.repeats)
        # Shuffle the repeated index list.
        perm = torch.randperm(indices.numel(), generator=generator)
        self.epoch += 1
        return iter(indices[perm].tolist())

    def __len__(self) -> int:
        # Return the number of samples in one training epoch.
        return self.dataset_size * self.repeats


class PhonemeDataset(Dataset):
    def __init__(
        self,
        condition: str,
        data_path: str,
    ) -> None:
        # Load the phoneme targets for this condition.
        df = pd.read_excel(str(data_path))
        df = df[df["condition"] == condition]
        self.phonemes = df["UR"].astype(str).tolist()
        self.targets = torch.tensor(df["target"].astype(float).tolist(), dtype=torch.float32)

        # Build the phoneme vocabulary.
        self.vocab = Vocab.build_vocab(
            symbols=self.phonemes,
            pad_id=hp.padding_id,
        )

    def __len__(self) -> int:
        return len(self.phonemes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return the encoded phoneme id and its target value.
        return {
            "x": torch.tensor(self.vocab.char_to_id[self.phonemes[idx]], dtype=torch.long),
            "y": self.targets[idx].clone(),
        }

    def get_collate_batch(self, augment_targets: bool):
        # Return the collate function for training or clean evaluation.
        def collate_batch(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            x = torch.stack([sample["x"] for sample in batch], dim=0)
            y = torch.stack([sample["y"] for sample in batch], dim=0)
            if augment_targets:
                y = add_noise(y)
            return {"x": x, "y": y}

        return collate_batch


class SourGrapeDataset(Dataset):
    def __init__(
        self,
        vocab: Vocab,
        condition: str,
        trajectory_data_path: str,
        trajectory_npy_root: str = hp.trajectory_npy_root,
        subset_seed: int = hp.seed,
        padding_value: float = hp.padding_value,
        max_trajectory_len: int = hp.max_trajectory_len,
    ) -> None:
        # Load the metadata for this condition.
        df = self._load_metadata(trajectory_data_path, condition)
        
        # Store the dataset item types.
        self.item_types = df["item_type"].tolist()
        self.pad_value = padding_value
        self.max_trajectory_len = max_trajectory_len
        
        # Resolve the trajectory directory.
        trajectory_root = Path(trajectory_npy_root).resolve()
        # Load the trajectory targets.
        sequences = self._load_trajectories(df["file_name"].tolist(), trajectory_root)

        # Encode each word as character ids.
        self.words = df["UR"].tolist()
        self.vocab = vocab
        encoded = [
            [self.vocab.char_to_id[ch] for ch in (w if isinstance(w, str) else "")]
            for w in self.words
        ]
        # Store the encoded words.
        self.x = torch.tensor(encoded, dtype=torch.long)
        # Store the original and current trajectory targets.
        self.y_real = [sequence.clone() for sequence in sequences]
        self.y_prev = [sequence.clone() for sequence in sequences]
        self.subsets = self._assign_subsets(subset_seed)

    def __len__(self) -> int:
        return len(self.words)

    def _load_metadata(self, data_path: str, condition: str) -> pd.DataFrame:
        # Load the rows for this condition.
        df = pd.read_csv(data_path)
        return df[df["condition"] == condition]

    def _load_trajectories(
        self,
        file_names: list[str],
        trajectory_root: Path,
    ) -> list[torch.Tensor]:
        # Load the trajectory targets.
        trajectories = []
        for rel_path in file_names:
            npy_path = trajectory_root / str(rel_path)
            arr = np.load(str(npy_path))
            flat = np.asarray(arr).reshape(-1)
            if len(flat) > self.max_trajectory_len:
                raise ValueError(
                    f"Trajectory length {len(flat)} exceeds max {self.max_trajectory_len}."
                )
            trajectories.append(torch.tensor(flat.astype(np.float32), dtype=torch.float32))
        return trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return the encoded word and its targets.
        return {
            "x": self.x[idx],
            "y_real": self.y_real[idx],
            "y_prev": self.y_prev[idx],
            "item_type": self.item_types[idx],
            "subset": self.subsets[idx],
        }

    def _assign_subsets(self, seed: int) -> list[str]:
        # Assign each item to subset a, b, or c.
        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randperm(len(self), generator=generator).tolist()
        split_size = len(self) // 3
        subset_labels = [""] * len(self)
        for idx in shuffled_indices[:split_size]:
            subset_labels[idx] = "a"
        for idx in shuffled_indices[split_size: split_size * 2]:
            subset_labels[idx] = "b"
        for idx in shuffled_indices[split_size * 2:]:
            subset_labels[idx] = "c"
        return subset_labels

    def pad_targets(self, targets: list[torch.Tensor]) -> torch.Tensor:
        # Pad the trajectory batch to a fixed-length tensor.
        padded = torch.full(
            (len(targets), self.max_trajectory_len),
            self.pad_value,
            dtype=torch.float32,
        )
        for idx, target in enumerate(targets):
            if len(target) > self.max_trajectory_len:
                raise ValueError(
                    f"Trajectory length {len(target)} exceeds max {self.max_trajectory_len}."
                )
            padded[idx, : len(target)] = target
        return padded

    def augment_targets(self, targets: list[torch.Tensor]) -> list[torch.Tensor]:
        # Augment the trajectory batch.
        augmented = []
        for target in targets:
            aug = augment_trajectory_variable_length(target.clone().unsqueeze(1)).squeeze(1)
            augmented.append(aug[: self.max_trajectory_len])
        return augmented

    def update_prev_targets(self, y_prev: torch.Tensor) -> None:
        # Update y_prev with the predicted trajectories.
        y_prev = y_prev.detach().cpu()
        self.y_prev = [
            y_prev[idx, : len(self.y_real[idx])].clone() for idx in range(len(self.y_real))
        ]

    def get_collate_batch(self, augment_targets: bool):
        # Return the collate function for this batching mode.
        def collate_batch(
            batch: list[Dict[str, torch.Tensor]],
        ) -> Dict[str, torch.Tensor | list[str] | list[int]]:
            # Stack the encoded word ids.
            x = torch.stack([sample["x"] for sample in batch], dim=0)
            if augment_targets:
                # Augment y_prev for this batch.
                targets = self.augment_targets([sample["y_prev"] for sample in batch])
            else:
                targets = [sample["y_prev"] for sample in batch]
            # Pad the batch targets.
            y_real = self.pad_targets([sample["y_real"] for sample in batch])
            y_prev = self.pad_targets(targets)
            return {
                "x": x,
                "y_real": y_real,
                "y_prev": y_prev,
                "item_type": [sample["item_type"] for sample in batch],
                "subset": [sample["subset"] for sample in batch],
            }

        return collate_batch
