# SourGrape: Velum Trajectory Mapping

This repo trains character-level models that map a word (e.g., a 5‑character UR string) to a fixed‑length articulatory trajectory (e.g., velum opening over time). Training runs in *generations*: each generation does phoneme pretraining to initialize embeddings, then trajectory training that uses the previous generation’s predictions as targets (`y_prev`). During each epoch, exposed-set and held-out-set evaluation are also measured against `y_prev`, while the final post-training evaluation is measured against the original trajectories (`y_real`).

The command-line entry point is `main.py`, which runs both conditions (`glide`, `fricative`) for a chosen number of iterations and generations.

## What The Code Does

1. **Phoneme pretraining** (`PhonemeRegressor` in `model.py`)
   - Reads `phoneme_target_file.xlsx` (UR → scalar target).
   - Trains an embedding + linear regressor to predict the scalar target.
   - Trains on the full phoneme set with `pretrain_repeats_per_epoch` repeated noisy passes per epoch.
   - Evaluates once per epoch on the same full phoneme set with no noise.
   - Saves the embedding plot, loss curves, and checkpoints.

2. **Trajectory training** (`LSTMRegressor` or `Seq2SeqRegressor` in `model.py`)
   - Reads one metadata file, `meta_file.csv`.
   - Loads each `.npy` trajectory, stores raw variable-length targets, and pads batches to `max_trajectory_len` during collation.
   - Uses pretrained phoneme embeddings and freezes them by default.
   - Assigns each item to one of three seeded subsets (`A`, `B`, `C`) when the trajectory dataset is loaded.
   - Builds:
     - a **training** loader over the exposed subsets
     - a **testing** loader over the exposed subsets
     - a **generalization** loader over the held-out subset
   - Trains a word→trajectory model and saves predictions + plots.
   - Records per-epoch `train`, `test`, and `gen` losses against `y_prev`.
   - Records one final `final` loss against `y_real` after training ends.
   - Updates `y_prev` row-by-row from the final predictions of the current generation.
   - Saves trajectory drift plots with mean curves and SD bands for exposed and held-out items.

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

To override model settings:

```bash
python main.py --model-type seq2seq
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

Current training setup: the workbook is expected to contain one row per phoneme target for each condition. The pretraining stage repeats that full set with fresh Gaussian noise during training and runs one clean `test` pass with no noise each epoch.

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

### 4) Repeated Training Passes
The phoneme pretraining stage uses the full phoneme set for both training and testing.

- During **pretraining**, each epoch repeats the full phoneme set `pretrain_repeats_per_epoch` times (default `500`) in mixed order using a sampler.
- During **pretraining testing**, each epoch iterates once over the same phoneme set with no noise.

The trajectory stage assigns one seeded three-way subset split for each condition.

- Generation `0` trains/tests on `A+B` and generalizes to `C`.
- Generation `1` trains/tests on `B+C` and generalizes to `A`.
- Generation `2` trains/tests on `A+C` and generalizes to `B`.
- During **training**, each epoch repeats the exposed subsets `train_repeats_per_epoch` times (default `20`) in mixed order using a sampler.
- During **testing**, each epoch iterates once over:
  - the exposed subsets
  - the held-out subset
- Because augmentation is applied during batch collation for training only, repeated appearances of the same item can receive different augmented versions of `y_prev` within the same epoch.

## Configuration

All hyperparameters live in `hyper_params.py`:

- Pretraining epochs and learning rate
- Model type: `model_type = "lstm"` or `"seq2seq"`
- Embedding size, hidden size, dropout, and teacher forcing ratio
- Training epochs and learning rate
- Data paths, repetition count, and padding values
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
      prediction_drift_test_<item_type>.png
      prediction_drift_gen_<item_type>.png
      loss_drift.png
      predictions.csv
    fricative_summary/
      pretrain_history.csv
      history.csv
      prediction_drift_test_<item_type>.png
      prediction_drift_gen_<item_type>.png
      loss_drift.png
      predictions.csv
  iteration_1/
    ...
```

Stage-specific behavior:

- `--stage all` writes both pretraining and trajectory-training artifacts.
- `--stage pretrain` writes per-generation pretraining artifacts plus `pretrain_history.csv` in each summary directory.
- `--stage train` skips phoneme pretraining, trains trajectory models without pretrained embeddings, and writes `history.csv`, `loss_drift.png`, trajectory drift plots, and `predictions.csv`.

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
- **Subset membership is stored per item**: each dataset item is assigned to subset `a`, `b`, or `c` when the trajectory dataset is loaded.
- **Padding is batch-time only**: trajectories are padded to a fixed length during collation and drift plotting. Longer trajectories stop the run by design.
- **Loss masking**: training and evaluation loss ignore padded values using `padding_value`.
- **Trajectory loss**: the trajectory stage currently uses masked MAE (`L1`) loss rather than masked MSE.
- **`y_prev` vs `y_real`**: per-epoch training, testing, and generalization use `y_prev` (previous generation predictions, with on-the-fly augmentation only during training), while the final post-training evaluation uses `y_real` (original targets).
- **Generation updates use true trajectory length**: when `y_prev` is updated after a generation, each prediction row is trimmed back to the original `y_real` length for that item.
- **Split rotation is cyclical**: the same seeded `A/B/C` split is reused across generations, while the exposed and held-out roles rotate.
- **Drift plots include variability**: the trajectory drift plots show mean trajectories with SD bands for the target and each generation.
- **`history.csv` / `pretrain_history.csv` columns**: each row stores the iteration, condition, generation, epoch, subset label, and loss value. Trajectory history rows use `train`, `test`, `gen`, and `final`; pretraining history rows use `train` and `test`.
- **`predictions.csv` columns**: each row stores the iteration, condition, generation, item index, word, item type, fixed subset label (`a`/`b`/`c`), scope label, and timestep values trimmed to the item's true trajectory length. Original targets are written with `generation = -1` and `scope = target`; model predictions use `scope = test` or `scope = gen`.
