from dataclasses import dataclass


@dataclass
class HyperParams:
    # Global configuration shared across stages.
    seed: int = 42  # change seed for individual iteration
    device: str = "cuda"  # "cuda" or "cpu"
    output_root: str = "output"

    # Data path configuration.
    phoneme_data_path: str = "dataset/phoneme_target_file.xlsx"
    trajectory_data_path: str = "dataset/meta_file.csv"
    trajectory_npy_root: str = "/mnt/storage/ldl_linguistics/SourGrape/raw_token_npy"
    penalty_data_path: str = "dataset/nasal_penalty_meta_file.csv"
    penalty_npy_root: str = "/mnt/storage/ldl_linguistics/SourGrape/nasal_penalty_npy"

    # Dataset configuration.
    pretrain_data_split_ratio = [0.8, 0.2]
    trajectory_pad_value: float = -999.0  # Padding value for trajectories.
    max_trajectory_len: int = 153
    pad_token_id: int = 0
    train_repeats_per_epoch: int = 20

    # Penalty loss configuration.
    penalty_loss_weight: float = 0.5
    penalty_threshold: float = 0.1  # Treat trajectory values above this as nasal activity.
    penalty_sigmoid_scale: float = 40.0  # Increase this to make the threshold sharper; decrease it to make the penalty signal softer.

    # Pretraining configuration.
    pretrain_epochs: int = 25
    pretrain_lr: float = 1e-4

    # Training configuration.
    batch_size: int = 16
    epochs: int = 25
    lr: float = 1e-4
    model_type: str = "lstm"  # "lstm" or "seq2seq"
    embed_size: int = 2
    hidden_size: int = 8
    num_layers: int = 1
    dropout: float = 0.5
