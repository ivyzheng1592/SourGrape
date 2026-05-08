# Global configuration shared across stages.
conditions = ["glide", "fricative"]
seed = 42  # change seed for individual iteration
device = "cuda"  # "cuda" or "cpu"
output_root = "output"
generations = 5
iterations = 5
stage = "all"  # "all", "pretrain", or "train"

# Data path configuration.
phoneme_data_path = "dataset/phoneme_target_file.xlsx"
trajectory_data_path = "dataset/meta_file.csv"
trajectory_npy_root = "/mnt/storage/ldl_linguistics/SourGrape/raw_token_npy"
penalty_data_path = "dataset/nasal_penalty_meta_file.csv"
penalty_npy_root = "/mnt/storage/ldl_linguistics/SourGrape/nasal_penalty_npy"

# Dataset configuration.
pretrain_data_split_ratio = [0.8, 0.2]
train_repeats_per_epoch = 20
max_trajectory_len = 153
padding_value = -999.0  # Padding value for trajectories.
padding_id = 0

# Penalty loss configuration.
penalty_loss_type = "sigmoid_bce"  # "sigmoid_bce", "relu_mse", or "softplus_mse"
penalty_loss_weight = 0.5
penalty_threshold = 0.1  # Treat trajectory values above this as nasal activity.
penalty_scale = 5.0  # Increase this to make the threshold sharper; decrease it to make the penalty signal softer.

# Pretraining configuration.
pretrain_epochs = 25
pretrain_lr = 5e-4

# Training configuration.
batch_size = 16
epochs = 25
lr = 1e-4
model_type = "lstm"  # "lstm" or "seq2seq"
bidirectional = False
embed_size = 2
hidden_size = 16
num_layers = 1
dropout = 0.5
teacher_forcing_ratio = 0.5
