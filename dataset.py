import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from hyper_params import HyperParams
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


class SourGrapeDataset(Dataset):
    def __init__(
        self,
        vocab: Vocab,
        condition: str,
        trajectory_data_path: str,
        trajectory_npy_root: str = HyperParams().trajectory_npy_root,
        penalty_data_path: str = HyperParams().penalty_data_path,
        penalty_npy_root: str = HyperParams().penalty_npy_root,
        trajectory_pad_value: float = HyperParams().trajectory_pad_value,
        max_trajectory_len: int = HyperParams().max_trajectory_len,
    ) -> None:
        # Load the metadata for this condition.
        df = self._load_metadata(trajectory_data_path, condition)
        penalty_df = self._load_metadata(penalty_data_path, condition)
        
        # Store the dataset item types.
        self.item_types = df["item_type"].tolist()
        self.pad_value = trajectory_pad_value
        self.max_trajectory_len = max_trajectory_len
        penalty_file_by_item_type = {
            row["item_type"]: row["file_name"]
            for _, row in penalty_df.iterrows()
        }
        
        # Resolve the trajectory and penalty directories.
        trajectory_root = Path(trajectory_npy_root).resolve()
        penalty_root = Path(penalty_npy_root).resolve()
        # Load the trajectory and penalty targets.
        sequences = self._load_trajectories(df["file_name"].tolist(), trajectory_root)
        penalty_targets = self._load_penalty_targets(
            item_types=self.item_types,
            penalty_file_by_item_type=penalty_file_by_item_type,
            penalty_root=penalty_root,
            trajectories=sequences,
        )

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
        self.penalty_targets = [target.clone() for target in penalty_targets]

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

    def _load_penalty_targets(
        self,
        item_types: list[str],
        penalty_file_by_item_type: dict[str, str],
        penalty_root: Path,
        trajectories: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Load the penalty targets.
        penalty_targets = []
        for item_type, trajectory in zip(item_types, trajectories):
            penalty_file = penalty_file_by_item_type[item_type]
            penalty_path = penalty_root / str(penalty_file)
            penalty_arr = np.load(str(penalty_path)).reshape(-1).astype(np.float32)
            if len(penalty_arr) != len(trajectory):
                raise ValueError(
                    f"Penalty length {len(penalty_arr)} does not match trajectory length {len(trajectory)}."
                )
            penalty_targets.append(torch.tensor(penalty_arr, dtype=torch.float32))
        return penalty_targets

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return the encoded word and its targets.
        return {
            "x": self.x[idx],
            "y_real": self.y_real[idx],
            "y_prev": self.y_prev[idx],
            "penalty_target": self.penalty_targets[idx],
            "item_type": self.item_types[idx],
        }

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
        if y_prev.shape[0] != len(self.y_prev):
            raise ValueError("y_prev must have the same number of rows as the dataset.")
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
            penalty_target = self.pad_targets(
                [sample["penalty_target"] for sample in batch]
            )
            return {
                "x": x,
                "y_real": y_real,
                "y_prev": y_prev,
                "penalty_target": penalty_target,
                "item_type": [sample["item_type"] for sample in batch],
            }

        return collate_batch


class PhonemeDataset(Dataset):
    def __init__(
        self,
        condition: str,
        data_path: str,
        augment: bool = False,
    ) -> None:
        # Load the phoneme targets for this condition.
        df = pd.read_excel(str(data_path))
        df = df[df["condition"] == condition]
        self.phonemes = df["UR"].astype(str).tolist()
        targets = torch.tensor(df["target"].astype(float).tolist(), dtype=torch.float32)
        
        # Add noise to the targets.
        if augment:
            targets = add_noise(targets)
        self.targets = targets.tolist()
        
        # Build the phoneme vocabulary.
        self.vocab = Vocab.build_vocab(
            symbols=self.phonemes,
            pad_id=HyperParams().pad_token_id,
        )

    def __len__(self) -> int:
        return len(self.phonemes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return the encoded phoneme id and its target value.
        return {
            "x": torch.tensor(self.vocab.char_to_id[self.phonemes[idx]], dtype=torch.long),
            "y": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
