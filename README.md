# LSTM Velum Trajectory Mapping

This project trains character-level models to map each input word to a fixed-length articulatory trajectory (e.g., velum opening over time), including multi-generation training.

## Data format (Metadata CSV)

Provide a `train_meta_file.csv` and `test_meta_file.csv` with one row per word.

Required columns:

- `UR`: the input word (must be exactly 5 letters).
- `file_name`: relative path to a `.npy` file containing the output trajectory.
- `condition`: condition label used to filter rows.
- `item_type`: item type label used for plotting summaries.

Each `.npy` file must contain **122 numbers** (the output trajectory). If a word is not
exactly 5 letters or a `.npy` file is not length 122, the loader warns and raises an error.

Example header:

```
UR,file_name,condition,item_type
```

## Quick start

1. Create a virtual environment and install requirements:

```
pip install -r requirements.txt
```

2. Example usage (single batch, device + loss):

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SourGrapeDataset
from model import LSTMRegressor, Seq2SeqRegressor

dataset = SourGrapeDataset(condition="glide", data_path="dataset/train_meta_file.csv", augment=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = next(iter(loader))
x = batch["x"].to(device)
y_prev = batch["y_prev"].to(device)
y_real = batch["y_real"].to(device)

model = LSTMRegressor(
    input_size=len(dataset.vocab),
    output_size=dataset.trajectory_len,
).to(device)
pred = model(x)
loss = nn.MSELoss()(pred, y_prev)
print(pred.shape, y_prev.shape, loss.item())

# Alternate model: simple seq2seq encoder-decoder
seq2seq = Seq2SeqRegressor(
    input_size=len(dataset.vocab),
    output_len=dataset.trajectory_len,
).to(device)
seq_pred = seq2seq(x)
seq_loss = nn.MSELoss()(seq_pred, y_prev)
print(seq_pred.shape, y_prev.shape, seq_loss.item())
```

3. Train multi-generation:

```
python iteration.py
```

## Outputs

The training run writes:

- `output/gen_<n>/models/`: checkpoints and final model for generation `n`
- `output/gen_<n>/predictions.npy`: predictions for generation `n`
- `output/gen_<n>/vocab.json`: character vocabulary
- `output/gen_<n>/history.csv`: training/test losses and final test loss
- `output/gen_<n>/loss_curve.png`: loss curve plot
- `output/gen_<n>/prediction_vs_target_<item_type>.png`: one plot per item type

## Notes

- Training uses `y_prev` targets, which are updated each generation with the previous model's predictions.
- Evaluation uses `y_real` targets from the metadata `.npy` files.
- The pipeline assumes `train_meta_file.csv`/`test_meta_file.csv` and `.npy` trajectories.
