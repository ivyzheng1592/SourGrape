from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def save_history_csv(
    iteration: int,
    condition: str,
    history_by_gen: Mapping[int, list[tuple[int, str, float]]],
    output_path: Path,
) -> None:
    # Save all history rows for all generations in one CSV file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("iteration,condition,generation,epoch,subset,loss\n")
        for gen, rows in history_by_gen.items():
            for epoch, subset, loss in rows:
                f.write(f"{iteration},{condition},{gen},{epoch},{subset},{loss}\n")


def save_loss_plot(history: dict, path: str) -> None:
    # Save a loss curve plot to disk.
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(6, 3))
    if train_loss:
        plt.plot(epochs, train_loss, label="train", linewidth=1.5)
    if test_loss:
        plt.plot(epochs, test_loss, label="test", linewidth=1.5)
    gen_loss = history.get("gen_loss", [])
    if gen_loss:
        plt.plot(range(1, len(gen_loss) + 1), gen_loss, label="gen", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_embedding_plot(
    embedding_weights: np.ndarray,
    id_to_char: Mapping[int, str],
    output_path: str,
) -> None:
    # Save a 2D PCA plot of the embedding matrix with character labels.
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embedding_weights)
    plt.figure(figsize=(5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, embedding_weights.shape[0]))
    plt.scatter(coords[:, 0], coords[:, 1], s=30, c=colors)
    for idx, (x, y) in enumerate(coords):
        label = id_to_char.get(idx, str(idx))
        plt.text(x, y, label, fontsize=9, ha="center", va="center", color="black")
        plt.scatter([], [], color=colors[idx], label=label)
    plt.title("Embedding PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_trajectory_plot(
    word: str,
    target: Sequence[float],
    prediction: Sequence[float],
    output_path: Path,
) -> None:
    # Save one prediction-vs-target trajectory plot for a single item.
    y_true = list(target)
    y_pred = list(prediction)
    x_true = list(range(len(y_true)))
    x_pred = list(range(len(y_pred)))
    plt.figure(figsize=(6, 3))
    plt.plot(x_true, y_true, label="target", linewidth=1.5)
    plt.plot(x_pred, y_pred, label="prediction", linewidth=1.5)
    plt.title(f"Prediction vs Target: {word}")
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_trajectory_plots(
    trajectory_dataset: Any,
    preds: np.ndarray,
    output_dir: Path,
    max_examples: int = 5,
) -> None:
    # Save one prediction plot per item type for a small sample of the dataset.
    seen_types = set()
    for idx in range(len(trajectory_dataset)):
        item_type = trajectory_dataset[idx]["item_type"]
        if item_type in seen_types:
            continue
        word = "".join(
            trajectory_dataset.vocab.id_to_char[i]
            for i in trajectory_dataset[idx]["x"].tolist()
        )
        target = trajectory_dataset[idx]["y_real"].tolist()
        prediction = preds[idx, : len(target)].tolist()
        pred_path = output_dir / f"prediction_{item_type}.png"
        save_trajectory_plot(word, target, prediction, pred_path)
        seen_types.add(item_type)
        if len(seen_types) >= max_examples:
            break


def save_trajectory_drift_plot(
    stats_by_gen: Mapping[int | str, Mapping[str, np.ndarray]],
    output_path: str,
) -> None:
    # Save one trajectory drift plot with SD bands for a single item type.
    gens = sorted(k for k in stats_by_gen.keys() if k != "target")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens)))

    plt.figure(figsize=(6, 3))
    if "target" in stats_by_gen:
        target_mean = stats_by_gen["target"]["mean"]
        target_std = stats_by_gen["target"]["std"]
        plt.plot(target_mean, color="black", linewidth=1.2, label="target")
        plt.fill_between(
            range(len(target_mean)),
            target_mean - target_std,
            target_mean + target_std,
            color="black",
            alpha=0.12,
        )
    for color, gen in zip(colors, gens):
        mean_traj = stats_by_gen[gen]["mean"]
        std_traj = stats_by_gen[gen]["std"]
        plt.plot(mean_traj, color=color, linewidth=1.0, label=f"gen_{gen}")
        plt.fill_between(
            range(len(mean_traj)),
            mean_traj - std_traj,
            mean_traj + std_traj,
            color=color,
            alpha=0.18,
        )
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.ylim(-0.25, 0.25)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_trajectory_drift_plots(
    summary_dir: Path,
    trajectory_dataset: Any,
    preds_by_gen: Mapping[int, np.ndarray],
    get_generation_labels: Callable[[int], tuple[set[str], str]],
    padding_value: float,
) -> None:
    item_types = list(trajectory_dataset.item_types)
    unique_types = sorted(set(item_types))
    # Pad all original targets into one matrix for summary statistics.
    targets = trajectory_dataset.pad_targets(trajectory_dataset.y_real).numpy()

    # Save separate exposed-set and held-out plots for each item type.
    for item_type in unique_types:
        stats_by_subset = {"test": {}, "gen": {}}
        target_stats_by_subset = {"test": None, "gen": None}

        for gen, preds in preds_by_gen.items():
            exposure_labels, heldout_label = get_generation_labels(gen)
            test_indices = []
            gen_indices = []
            # Collect indices for the exposed test subset and held-out gen subset.
            for idx, subset in enumerate(trajectory_dataset.subsets):
                if item_types[idx] != item_type:
                    continue
                if subset in exposure_labels:
                    test_indices.append(idx)
                elif subset == heldout_label:
                    gen_indices.append(idx)

            # Collect predictions and targets for each subset
            for subset_name, indices in (("test", test_indices), ("gen", gen_indices)):
                if not indices:
                    raise ValueError(
                        f"No items found for item_type={item_type!r}, "
                        f"subset={subset_name!r}, generation={gen}."
                    )
                targets_subset = targets[indices]
                preds_subset = preds[indices]
                # Mask padded timesteps before computing target and prediction summaries.
                mask = targets_subset != padding_value
                masked_preds = np.where(mask, preds_subset, np.nan)
                masked_targets = np.where(mask, targets_subset, np.nan)
                if target_stats_by_subset[subset_name] is None:
                    target_stats_by_subset[subset_name] = {
                        "mean": np.nanmean(masked_targets, axis=0),
                        "std": np.nanstd(masked_targets, axis=0),
                    }
                stats_by_subset[subset_name][gen] = {
                    "mean": np.nanmean(masked_preds, axis=0),
                    "std": np.nanstd(masked_preds, axis=0),
                }

        # Plot the generation summaries for each subset.
        for subset_name in ("test", "gen"):
            stats_by_gen = stats_by_subset[subset_name]
            target_stats = target_stats_by_subset[subset_name]
            if target_stats is not None:
                stats_by_gen["target"] = target_stats
            trajectory_drift_path = summary_dir / f"prediction_drift_{subset_name}_{item_type}.png"
            save_trajectory_drift_plot(stats_by_gen, str(trajectory_drift_path))


def save_loss_drift_plot(
    train_history_by_gen: Mapping[int, list[tuple[int, str, float]]],
    output_path: str,
) -> None:
    # Save train/test/gen loss curves across generations.
    gens = sorted(train_history_by_gen.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens)))
    subset_names = ("train", "test", "gen")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(subset_names),
        figsize=(5 * len(subset_names), 3),
        sharex=False,
        sharey=False,
    )

    # Plot one loss panel per subset, with one line per generation.
    for ax, subset_name in zip(axes, subset_names):
        for color, gen in zip(colors, gens):
            rows = [
                (epoch, float(loss))
                for epoch, subset, loss in train_history_by_gen[gen]
                if subset == subset_name
            ]
            if not rows:
                continue
            epochs = [epoch for epoch, _ in rows]
            values = [loss for _, loss in rows]
            ax.plot(
                epochs,
                values,
                color=color,
                linewidth=1.2,
                label=f"gen_{gen}",
            )
        ax.set_title(f"{subset_name} by Generation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_predictions_csv(
    iteration: int,
    condition: str,
    targets: Sequence[Sequence[float]],
    preds_by_gen: Mapping[int, np.ndarray],
    words: Sequence[str],
    item_types: Sequence[str],
    subsets: Sequence[str],
    get_generation_labels: Callable[[int], tuple[set[str], str]],
    output_path: Path,
) -> None:
    # Save all generation predictions in one CSV file.
    target_lengths = [len(target) for target in targets]
    max_len = max(target_lengths)
    timestep_cols = ",".join(f"timestep_{idx}" for idx in range(max_len))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"iteration,condition,generation,item_index,word,item_type,subset,scope,"
            f"{timestep_cols}\n"
        )
        for idx, target in enumerate(targets):
            valid_target = list(target)
            padded_target = valid_target + [""] * (max_len - len(valid_target))
            target_values = ",".join(str(value) for value in padded_target)
            f.write(
                f"{iteration},{condition},-1,{idx},{words[idx]},{item_types[idx]},"
                f"{subsets[idx]},target,{target_values}\n"
            )
        for gen, preds in preds_by_gen.items():
            # Label each item as exposed-set test data or held-out gen data for this generation.
            exposure_labels, _ = get_generation_labels(gen)
            for idx, pred in enumerate(preds):
                # Trim each prediction back to the item's true trajectory length.
                valid_pred = pred[: target_lengths[idx]]
                padded_pred = list(valid_pred) + [""] * (max_len - len(valid_pred))
                pred_values = ",".join(str(value) for value in padded_pred)
                scope_name = "test" if subsets[idx] in exposure_labels else "gen"
                # Write one row with item metadata and timestep values.
                f.write(
                    f"{iteration},{condition},{gen},{idx},{words[idx]},{item_types[idx]},"
                    f"{subsets[idx]},{scope_name},{pred_values}\n"
                )
