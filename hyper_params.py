from dataclasses import dataclass


@dataclass
class HyperParams:
    # Global configuration shared across stages.
    seed: int = 42  # change seed for individual iteration
    device: str = "cuda"  # "cuda" or "cpu"
    output_root: str = "output"

    # Dataset configuration for phoneme and trajectory stages.
    phoneme_data_path: str = "dataset/phoneme_target_file.xlsx"
    data_split_ratio = [0.8, 0.2]
    train_data_path: str = "dataset/train_meta_file.csv"
    test_data_path: str = "dataset/test_meta_file.csv"
    npy_root: str = "dataset"
    trajectory_pad_value: float = -999.0  # Padding value for trajectories.
    max_trajectory_len: int = 153
    pad_token_id: int = 0

    # Pretraining configuration.
    pretrain_epochs: int = 5
    pretrain_lr: float = 1e-4

    # Training configuration.
    batch_size: int = 16
    epochs: int = 25
    lr: float = 1e-4
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    embed_size: int = 4
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.5
