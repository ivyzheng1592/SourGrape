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
    data_path: str = "dataset/jitter_meta_file.csv"
    output_root: str = "output"
    embed_size: int = 16
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.5
    expected_word_len: int = 5
    expected_trajectory_len: int = 122
