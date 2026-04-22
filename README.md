# SourGrape: Velum Trajectory Mapping

This repo trains character-level models that map a word (e.g., a 5‑character UR string) to a fixed‑length articulatory trajectory (e.g., velum opening over time). Training runs in *generations*: each generation does phoneme pretraining to initialize embeddings, then trajectory training that uses the previous generation’s predictions as targets (`y_prev`) while always evaluating against the original trajectories (`y_real`).

The command-line entry point is `main.py`, which can run one or both conditions (`glide`, `fricative`) for a chosen number of generations.

## What The Code Does

1. **Phoneme pretraining** (`PhonemeRegressor` in `model.py`)
   - Reads `phoneme_target_file.xlsx` (UR → scalar target).
   - Trains an embedding + linear regressor to predict the scalar target.
   - Saves embedding PCA plot, loss curves, and checkpoints.

2. **Trajectory training** (`LSTMRegressor` or `Seq2SeqRegressor` in `model.py`)
   - Reads one metadata file, `meta_file.csv`.
   - Loads each `.npy` trajectory, stores raw variable-length targets, and pads batches to `max_trajectory_len` during collation.
   - Loads a penalty target for each `item_type` from `nasal_penalty_meta_file.csv`.
   - Builds two dataloaders from the same dataset:
     - a **training** loader that sees each item `train_repeats_per_epoch` times per epoch in mixed order, with on-the-fly augmentation applied to `y_prev`
     - a **testing** loader that sees each item once per epoch with no augmentation
   - Trains a word→trajectory model and saves predictions + plots.
   - Updates `y_prev` row-by-row from the final predictions of the current generation.
   - Saves trajectory drift plots with mean curves and SD bands across generations.
   - Adds an auxiliary penalty loss that compares predicted nasal activity against the penalty target.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your data under `dataset/` (see “Data requirements” below).

3. Run multi‑generation training:

```bash
python main.py --seed 42
```

To run a specific condition or change the number of generations:

```bash
python main.py --condition glide --generations 3 --seed 42
```

To run only one stage:

```bash
python main.py --condition glide --stage pretrain --seed 42
python main.py --condition glide --stage train --seed 42
```

## Data Requirements

### 1) Metadata CSV
`hyper_params.py` expects:

- `dataset/meta_file.csv`

Required columns (the code reads these exact names):

- `UR`: the input word string used for character encoding.
- `file_name`: relative path to a `.npy` trajectory file (resolved under `dataset/`).
- `condition`: label used to filter rows (e.g., `glide`, `fricative`).
- `item_type`: used to group plots.

Note: the CSV in this repo also includes a `word` column; it is currently **not used** by the code.

**Word length requirement**: all `UR` strings must be **exactly 5 letters**. The dataset encodes `UR` directly into a fixed-size tensor, so any other length will break batching.

### 2) Trajectory `.npy` files
Each `file_name` should point to a `.npy` file containing a 1D or flattenable trajectory. The raw data may be multi‑dimensional for use in other projects, but **this project flattens it**. The code:

- Flattens each array
- Stores the raw trajectory in the dataset
- Pads batches to `max_trajectory_len` (default `153`) using `trajectory_pad_value` (default `-999.0`)
- **Raises an error** if any trajectory is longer than `max_trajectory_len` (this is intentional; longer trajectories indicate a data problem)

### 3) Phoneme metadata XLSX
`hyper_params.py` expects:

- `dataset/phoneme_target_file.xlsx`

Required columns:

- `UR`: a single character (e.g., `m`)
- `target`: scalar target for pretraining
- `condition`: label used to filter rows (e.g., `glide`, `fricative`)

Note: the character vocabulary is built from the phoneme dataset for each condition, so all characters in `UR` must appear in the phoneme file for that same condition.

### 4) Nasal Penalty Metadata
`hyper_params.py` expects:

- `dataset/nasal_penalty_meta_file.csv`
- `dataset/nasal_penalty/`

Required columns:

- `item_type`: item type in the main metadata file
- `condition`: condition label used to filter rows
- `file_name`: penalty `.npy` file for that item type

Each penalty file is a time-aligned target used by the auxiliary penalty loss during trajectory training.

### 5) Repeated Training Passes
The trajectory stage uses a single dataset for both training and testing.

- During **training**, each epoch repeats the full dataset `train_repeats_per_epoch` times (default `20`) in mixed order using a sampler.
- During **testing**, each epoch iterates over the same dataset once.
- Because augmentation is applied during batch collation for training only, repeated appearances of the same item can receive different augmented versions of `y_prev` within the same epoch.

## Configuration

All hyperparameters live in `hyper_params.py`:

- Model type: `model_type = "lstm"` or `"seq2seq"`
- Embedding size, hidden size, dropout, etc.
- Training epochs and learning rates
- Data path, repetition count, and padding values
- Penalty loss paths and parameters
- Device (`cpu`/`cuda`)

## Outputs

Each run writes to:

```
output/<condition>_seed<seed>_<timestamp>/
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

If `--stage pretrain` is used, the run only writes pretraining artifacts.
If `--stage train` is used, the run skips phoneme pretraining and trains the trajectory model without pretrained embeddings.

## Repository Tour

- `main.py`: command-line entry point
- `iteration.py`: multi‑generation training loop
- `dataset.py`: dataset classes + vocab handling
- `model.py`: LSTM/seq2seq regressors + phoneme regressor
- `train_eval.py`: training/evaluation loops
- `preprocessing.py`: trajectory augmentation
- `utils.py`: plotting helpers
- `hyper_params.py`: all configuration

## Trajectory Handling Notes

- **Flattening is intentional**: trajectories are flattened so the model predicts a single 1D vector, even if the raw data is multi‑dimensional.
- **Raw trajectories are stored**: `y_real` and `y_prev` are kept as variable-length trajectories inside the dataset.
- **Padding is batch-time only**: trajectories are padded to a fixed length during collation and drift plotting. Longer trajectories stop the run by design.
- **Loss masking**: training and evaluation loss ignore padded values using `trajectory_pad_value`.
- **`y_prev` vs `y_real`**: training uses `y_prev` (previous generation predictions, with on-the-fly augmentation), while evaluation uses `y_real` (original targets).
- **Generation updates use true trajectory length**: when `y_prev` is updated after a generation, each prediction row is trimmed back to the original `y_real` length for that item.
- **Drift plots include variability**: the mean trajectory drift plot shows mean trajectories with SD bands for the target and each generation.
- **Penalty loss supervision**: trajectory training also compares predicted nasal activity against the penalty targets loaded from `nasal_penalty_meta_file.csv`.

## Other Notes

- Phoneme pretraining embeddings are passed into the trajectory model and frozen by default.
- Train-time trajectory augmentation is applied on the fly during batch collation, not when the dataset is first loaded.
- Phoneme targets can be perturbed with Gaussian noise when building the dataset (`augment=True`).
