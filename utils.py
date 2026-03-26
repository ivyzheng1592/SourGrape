from typing import Iterable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_mean_trajectory_drift(
    preds_by_gen: Mapping[int, np.ndarray],
    output_path: str,
) -> None:
    # Save one mean trajectory drift plot for a single item type.
    if not preds_by_gen:
        return
    gens = sorted(k for k in preds_by_gen.keys() if k != "target")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens)))

    plt.figure(figsize=(6, 3))
    # Use "target" key if provided.
    if "target" in preds_by_gen:
        plt.plot(preds_by_gen["target"], color="black", linewidth=1.2, label="target")
    for color, gen in zip(colors, gens):
        mean_traj = preds_by_gen[gen]
        plt.plot(mean_traj, color=color, linewidth=1.0, label=f"gen_{gen}")
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_drift(
    history_by_gen: Mapping[int, dict],
    output_path: str,
) -> None:
    # Save training/test loss curves across generations in two subplots.
    if not history_by_gen:
        return
    gens = sorted(history_by_gen.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens)))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), sharex=False, sharey=False)
    ax_train, ax_test = axes

    for color, gen in zip(colors, gens):
        hist = history_by_gen[gen]
        train_loss = hist.get("train_loss", [])
        test_loss = hist.get("test_loss", [])
        if train_loss:
            ax_train.plot(
                range(1, len(train_loss) + 1),
                train_loss,
                color=color,
                linewidth=1.2,
                label=f"gen_{gen}",
            )
        if test_loss:
            ax_test.plot(
                range(1, len(test_loss) + 1),
                test_loss,
                color=color,
                linewidth=1.2,
                label=f"gen_{gen}",
            )

    ax_train.set_title("Train Loss by Generation")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_test.set_title("Test Loss by Generation")
    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Loss")
    ax_train.legend()
    ax_test.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_embedding_pca(
    embedding_weights: np.ndarray,
    id_to_char: Mapping[int, str],
    output_path: str,
) -> None:
    # Save a 2D PCA plot of the embedding matrix with character labels.
    if embedding_weights.ndim != 2 or embedding_weights.shape[0] == 0:
        return
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embedding_weights)
    plt.figure(figsize=(5, 5))
    plt.scatter(coords[:, 0], coords[:, 1], s=30)
    for idx, (x, y) in enumerate(coords):
        label = id_to_char.get(idx, str(idx))
        plt.text(x, y, label, fontsize=9, ha="center", va="center")
    plt.title("Embedding PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
