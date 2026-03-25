from dataclasses import dataclass


@dataclass
class HyperParams:
    # Training configuration.
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    data_split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    device: str = "cuda"  # "cuda" or "cpu"
