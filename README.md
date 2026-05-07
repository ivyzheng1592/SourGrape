# SourGrape: Velum Trajectory Mapping

This repo trains character-level models that map a word (e.g., a 5‑character UR string) to a fixed‑length articulatory trajectory (e.g., velum opening over time). Training runs in *generations*: each generation does phoneme pretraining to initialize embeddings, then trajectory training that uses the previous generation’s predictions as targets (`y_prev`) while always evaluating against the original trajectories (`y_real`).

The command-line entry point is `main.py`, which runs both conditions (`glide`, `fricative`) for a chosen number of iterations and generations.

## What The Code Does

1. **Phoneme pretraining** (`PhonemeRegressor` in `model.py`)
   - Reads `phoneme_target_file.xlsx` (UR → scalar target).
   - Trains an embedding + linear regressor to predict the scalar target.
   - Can perturb phoneme targets with Gaussian noise when building the dataset (`augment=True`).
   - Saves the embedding plot, loss curves, and checkpoints.

2. **Trajectory training** (`LSTMRegressor` or `Seq2SeqRegressor` in `model.py`)
   - Reads one metadata file, `meta_file.csv`.
   - Loads each `.npy` trajectory, stores raw variable-length targets, and pads batches to `max_trajectory_len` during collation.
   - Loads a penalty target for each `item_type` from `nasal_penalty_meta_file.csv`.
   - Uses pretrained phoneme embeddings and freezes them by default.
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
python main.py
```

To change the number of generations:

```bash
python main.py --generations 3
```

To run only one stage:

```bash
python main.py --stage pretrain
python main.py --stage train
```

To override model and penalty-loss settings:

```bash
python main.py --model-type seq2seq --penalty-loss-type relu_mse --penalty-loss-weight 1.0
```

## Data Requirements

### 1) Phoneme metadata XLSX
`hyper_params.py` expects:

- `dataset/phoneme_target_file.xlsx`

Required columns:

- `UR`: a single character (e.g., `m`)
- `target`: scalar target for pretraining
- `condition`: label used to filter rows (e.g., `glide`, `fricative`)

Note: the character vocabulary is built from the phoneme dataset for each condition, so all characters in `UR` must appear in the phoneme file for that same condition.

### 2) Metadata CSV
`hyper_params.py` expects:

- `dataset/meta_file.csv`

Required columns (the code reads these exact names):

- `UR`: the input word string used for character encoding.
- `file_name`: relative path to a `.npy` trajectory file (resolved under `dataset/`).
- `condition`: label used to filter rows (e.g., `glide`, `fricative`).
- `item_type`: used to group plots.

Note: the CSV in this repo also includes a `word` column; it is currently **not used** by the code.

**Word length requirement**: all `UR` strings must be **exactly 5 letters**. The dataset encodes `UR` directly into a fixed-size tensor, so any other length will break batching.

### 3) Trajectory `.npy` files
Each `file_name` should point to a `.npy` file containing a 1D or flattenable trajectory. The raw data may be multi‑dimensional for use in other projects, but **this project flattens it**. The code:

- Flattens each array
- Stores the raw trajectory in the dataset
- Pads batches to `max_trajectory_len` (default `153`) using `padding_value` (default `-999.0`)
- **Raises an error** if any trajectory is longer than `max_trajectory_len` (this is intentional; longer trajectories indicate a data problem)

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

- Pretraining epochs and learning rate
- Model type: `model_type = "lstm"` or `"seq2seq"`
- Embedding size, hidden size, dropout, and teacher forcing ratio
- Training epochs and learning rate
- Data paths, repetition count, and padding values
- Penalty loss paths and parameters
- Device (`cpu`/`cuda`)

## Outputs

Each call to `run_iterations()` writes to:

```
output/iterations_<timestamp>/
  run_config.txt
  iteration_0/
    glide_gen_0/
      pretrain_models/
      pretrain_loss_curve.png
      pretrain_embedding.png
      models/
      loss_curve.png
      prediction_<item_type>.png
    fricative_gen_0/
      ...
    glide_summary/
      pretrain_history.csv
      history.csv
      prediction_drift_<item_type>.png
      loss_drift.png
      predictions.csv
    fricative_summary/
      pretrain_history.csv
      history.csv
      prediction_drift_<item_type>.png
      loss_drift.png
      predictions.csv
  iteration_1/
    ...
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
- **Loss masking**: training and evaluation loss ignore padded values using `padding_value`.
- **`y_prev` vs `y_real`**: training uses `y_prev` (previous generation predictions, with on-the-fly augmentation), while evaluation uses `y_real` (original targets).
- **Generation updates use true trajectory length**: when `y_prev` is updated after a generation, each prediction row is trimmed back to the original `y_real` length for that item.
- **Drift plots include variability**: the trajectory drift plot shows mean trajectories with SD bands for the target and each generation.
- **Penalty loss supervision**: trajectory training also compares predicted nasal activity against the penalty targets loaded from `nasal_penalty_meta_file.csv`.
