# Global configuration shared across stages.
patterns = ["AH", "SG"]
conditions = ["glide", "fricative"]
seed = 1234  # change seed for individual iteration
device = "cuda"  # "cuda" or "cpu"
output_root = "output"
generations = 5
iterations = 5
stage = "all"  # "all", "pretrain", or "train"

# Data path configuration.
phoneme_data_path = "dataset/phoneme_target_file.xlsx"
trajectory_data_path = "dataset/meta_file_ahsg.csv"
trajectory_npy_root = "/mnt/storage/ldl_linguistics/SourGrape/raw_token_npy"

# Dataset configuration.
pretrain_repeats_per_epoch = 500
train_repeats_per_epoch = 20
max_trajectory_len = 153
padding_value = -999.0  # Padding value for trajectories.
padding_id = 0

# Pretraining configuration.
pretrain_epochs = 25
pretrain_lr = 5e-4

# Training configuration.
batch_size = 4
epochs = 25
lr = 1e-4
model_type = "lstm"  # "lstm" or "seq2seq"
embed_size = 2
hidden_size = 16
num_layers = 1
dropout = 0.2
teacher_forcing_ratio = 0.5
