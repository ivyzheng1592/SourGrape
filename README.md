# LSTM Velum Trajectory Mapping

This project trains a character-level LSTM to map each input word to a fixed-length articulatory trajectory (e.g., velum opening over time).

## Data format (Metadata CSV)

Provide a `jitter_meta_file.csv` with one row per word.

Required columns:

- `UR`: the input word (must be exactly 5 letters).
- `jitter_filename`: relative path to a `.npy` file containing the output trajectory.

Each `.npy` file must contain **122 numbers** (the output trajectory). If a word is not
exactly 5 letters or a `.npy` file is not length 122, the loader prints a warning and
pads/truncates to length 122 for training stability.

Example header:

```
UR,jitter_filename
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

dataset = SourGrapeDataset(condition="glide")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = next(iter(loader))
x, y = batch["x"].to(device), batch["y"].to(device)

model = LSTMRegressor(
    input_size=len(dataset.vocab),
    output_size=dataset.trajectory_len,
).to(device)
pred = model(x)
loss = nn.MSELoss()(pred, y)
print(pred.shape, y.shape, loss.item())

# Alternate model: simple seq2seq encoder-decoder
seq2seq = Seq2SeqRegressor(
    input_size=len(dataset.vocab),
    output_len=dataset.trajectory_len,
).to(device)
seq_pred = seq2seq(x)
seq_loss = nn.MSELoss()(seq_pred, y)
print(seq_pred.shape, y.shape, seq_loss.item())
```

3. Train:

```
python train.py --data_path dataset/jitter_meta_file.csv --output_dir artifacts
```

## Outputs

The training run writes:

- `artifacts/model.pt`: model state dict
- `artifacts/vocab.json`: character vocabulary
- `artifacts/metrics.json`: final metrics and config

## Notes

- This baseline maps each word to a fixed-length trajectory. If your trajectories are variable length, we can upgrade the model to a sequence-to-sequence setup.
- The pipeline assumes a CSV metadata file and `.npy` trajectories.
