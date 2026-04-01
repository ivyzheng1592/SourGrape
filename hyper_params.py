from dataclasses import dataclass


@dataclass
class HyperParams:
    # Training configuration.
    batch_size: int = 16
    epochs: int = 2
    lr: float = 1e-4
    seed: int = 42  # change seed for individual iteration
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    device: str = "cuda"  # "cuda" or "cpu"
    train_data_path: str = "dataset/train_meta_file.csv"
    test_data_path: str = "dataset/test_meta_file.csv"
    output_root: str = "output"
    embed_size: int = 4
    hidden_size: int = 8
    num_layers: int = 1
    dropout: float = 0.5
    expected_word_len: int = 5
    trajectory_pad_value: float = -999.0
    max_trajectory_len: int = 153
