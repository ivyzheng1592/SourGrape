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
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_mean_trajectory_drift(
    preds_by_gen: Mapping[int, np.ndarray],
    item_types: Sequence[str],
    output_dir: str,
) -> None:
    # Save mean trajectory plots per item type across generations.
    if not preds_by_gen:
        return
    item_types = list(item_types)
    unique_types = sorted(set(item_types))
    for item_type in unique_types:
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        plt.figure(figsize=(6, 3))
        for gen in sorted(preds_by_gen.keys()):
            preds = preds_by_gen[gen]
            mean_traj = preds[idxs].mean(axis=0)
            plt.plot(mean_traj, label=f"gen_{gen}", linewidth=1.5)
        plt.title(f"Mean Trajectory Drift: {item_type}")
        plt.xlabel("Time index")
        plt.ylabel("Trajectory value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mean_trajectory_{item_type}.png")
        plt.close()
