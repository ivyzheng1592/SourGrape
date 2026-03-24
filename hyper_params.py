from dataclasses import dataclass


@dataclass
class HyperParams:
    # Training configuration.
    condition: str = "glide"
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    data_split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    resume_path: str = ""  # Path to a checkpoint .pt file, or "" to start fresh.
