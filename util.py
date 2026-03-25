from typing import Iterable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_word_trajectory(word: str, trajectory: Sequence[float], title_prefix: str = "Input") -> None:
    # Plot a single word and its trajectory for quick inspection.
    y = list(trajectory)
    x = list(range(len(y)))
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, linewidth=1.5)
    plt.title(f"{title_prefix}: {word}")
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.tight_layout()
    plt.show()


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
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_prediction_plot(
    word: str,
    target: Sequence[float],
    prediction: Sequence[float],
    path: str,
    title_prefix: str = "Prediction vs Target",
) -> None:
    # Save a predicted vs target trajectory plot to disk.
    y_true = list(target)
    y_pred = list(prediction)
    x_true = list(range(len(y_true)))
    x_pred = list(range(len(y_pred)))
    plt.figure(figsize=(6, 3))
    plt.plot(x_true, y_true, label="target", linewidth=1.5)
    plt.plot(x_pred, y_pred, label="prediction", linewidth=1.5, linestyle="--")
    plt.title(f"{title_prefix}: {word}")
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.ylim(-0.10, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_mean_trajectory_drift(
    preds_by_gen: Mapping[int, np.ndarray],
    item_types: Sequence[str],
    output_path: str,
    targets: np.ndarray,
) -> None:
    # Save a grid of mean trajectory drift plots across item types.
    if not preds_by_gen:
        return
    item_types = list(item_types)
    unique_types = sorted(set(item_types))
    n_types = len(unique_types)
    ncols = 3
    nrows = (n_types + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    gens = sorted(preds_by_gen.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens)))

    for ax_idx, item_type in enumerate(unique_types):
        ax = axes[ax_idx]
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        # Plot target mean.
        target_mean = targets[idxs].mean(axis=0)
        ax.plot(target_mean, color="black", linewidth=1.5, label="target")
        # Plot generation means with continuous color scale.
        for color, gen in zip(colors, gens):
            preds = preds_by_gen[gen]
            mean_traj = preds[idxs].mean(axis=0)
            ax.plot(mean_traj, color=color, linewidth=1.2, label=f"gen_{gen}")
        ax.set_title(item_type)

    # Hide any extra axes.
    for ax in axes[n_types:]:
        ax.axis("off")

    fig.suptitle("Mean Trajectory Drift by Item Type")
    fig.supxlabel("Time index")
    fig.supylabel("Trajectory value")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
