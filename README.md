# SourGrape: Velum Trajectory Mapping

This repo trains character-level models that map a word (e.g., a 5‑character UR string) to a fixed‑length articulatory trajectory (e.g., velum opening over time). Training runs in *generations*: each generation does phoneme pretraining to initialize embeddings, then trajectory training that uses the previous generation’s predictions as targets (`y_prev`).

The main entry point is `iteration.py`, which runs two conditions (`glide`, `fricative`) for 5 generations each.

## What The Code Does

1. **Phoneme pretraining** (`PhonemeRegressor` in `model.py`)
   - Reads `phoneme_target_file.xlsx` (UR → scalar target).
   - Trains an embedding + linear regressor to predict the scalar target.
   - Saves embedding PCA plot, loss curves, and checkpoints.

2. **Trajectory training** (`LSTMRegressor` or `Seq2SeqRegressor` in `model.py`)
   - Reads `train_meta_file.csv` and `test_meta_file.csv`.
   - Loads each `.npy` trajectory, pads to `max_trajectory_len`, and (optionally) augments train trajectories.
   - Trains a word→trajectory model and saves predictions + plots.
   - Updates `y_prev` in the training set using **test‑set predictions matched by word** (see “Data requirements”).

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your data under `dataset/` (see “Data requirements” below).

3. Run multi‑generation training:

```bash
python iteration.py
```

## Data Requirements

### 1) Train/Test metadata CSVs
`hyper_params.py` expects:

- `dataset/train_meta_file.csv`
- `dataset/test_meta_file.csv`

Required columns (the code reads these exact names):

- `UR`: the input word string used for character encoding.
- `file_name`: relative path to a `.npy` trajectory file (resolved under `dataset/`).
- `condition`: label used to filter rows (e.g., `glide`, `fricative`).
- `item_type`: used to group plots.

Note: the CSVs in this repo also include a `word` column; it is currently **not used** by the code.

**Word length requirement**: all `UR` strings must be **exactly 5 letters**. The dataset encodes `UR` directly into a fixed-size tensor, so any other length will break batching.

### 2) Trajectory `.npy` files
Each `file_name` should point to a `.npy` file containing a 1D or flattenable trajectory. The raw data may be multi‑dimensional for use in other projects, but **this project flattens it**. The code:

- Flattens each array
- Pads to `max_trajectory_len` (default `153`) using `trajectory_pad_value` (default `-999.0`)
- **Raises an error** if any trajectory is longer than `max_trajectory_len` (this is intentional; longer trajectories indicate a data problem)

### 3) Phoneme metadata XLSX
`hyper_params.py` expects:

- `dataset/phoneme_target_file.xlsx`

Required columns:

- `UR`: a single character (e.g., `m`)
- `target`: scalar target for pretraining
- `condition`: label used to filter rows (e.g., `glide`, `fricative`)

Note: the character vocabulary is built from the phoneme dataset for each condition, so all characters in `UR` must appear in the phoneme file for that same condition.


### 4) Train/Test coupling requirement
During generation updates, the code maps **test predictions to training items by word**. That means every training `UR` must appear in the **test** CSV. In practice, the training set is **n×** the testing set (n is controlled by how you build the CSVs), and each test `UR` is repeated across the training file. If any training word is missing from test, `run_trajectory_training` will raise:

```
ValueError: Missing prediction for word: ...
```

## Configuration

All hyperparameters live in `hyper_params.py`:

- Model type: `model_type = "lstm"` or `"seq2seq"`
- Embedding size, hidden size, dropout, etc.
- Training epochs and learning rates
- Data paths and padding values
- Device (`cpu`/`cuda`)

## Outputs

Each run writes to:

```
output/<condition>_<timestamp>/
  gen_<n>/
    pretrain_models/
    pretrain_history.csv
    pretrain_loss_curve.png
    embedding_pca.png
    models/
    history.csv
    loss_curve.png
    predictions.npy
    prediction_vs_target_<item_type>.png
  drift_plots/
    mean_drift_<item_type>.png
    loss_drift.png
```

## Repository Tour

- `iteration.py`: multi‑generation training loop (main entry point)
- `dataset.py`: dataset classes + vocab handling
- `model.py`: LSTM/seq2seq regressors + phoneme regressor
- `train_eval.py`: training/evaluation loops
- `preprocessing.py`: trajectory augmentation
- `utils.py`: plotting helpers
- `hyper_params.py`: all configuration

## Trajectory Handling Notes

- **Flattening is intentional**: trajectories are flattened so the model predicts a single 1D vector, even if the raw data is multi‑dimensional.
- **Padding is required**: trajectories are padded to a fixed length. Longer trajectories stop the run by design.
- **Loss masking**: training and evaluation loss ignore padded values using `trajectory_pad_value`.
- **`y_prev` vs `y_real`**: training uses `y_prev` (previous generation predictions), while evaluation uses `y_real` (original targets).

## Other Notes

- Phoneme pretraining embeddings are passed into the trajectory model and frozen by default.
- Augmentation is enabled for training trajectories by default (`augment=True`).
- Phoneme targets can be perturbed with Gaussian noise when building the dataset (`augment=True`).
