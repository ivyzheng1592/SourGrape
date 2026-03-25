from dataclasses import dataclass


@dataclass
class HyperParams:
    # Training configuration.
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    seed: int = 42
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    device: str = "cuda"  # "cuda" or "cpu"
