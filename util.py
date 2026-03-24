from typing import Iterable, Sequence

import matplotlib.pyplot as plt


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


def plot_batch_sample(words: Iterable[str], trajectories: Iterable[Sequence[float]]) -> None:
    # Plot the first sample from a batch of words/trajectories.
    for word, traj in zip(words, trajectories):
        plot_word_trajectory(word, traj)
        break


def plot_dataset_sample(dataset, idx: int = 0, title_prefix: str = "Input") -> None:
    # Plot a single sample from SourGrapeDataset.
    item = dataset[idx]
    x = item["x"].tolist()
    # Decode ids back to a word string for display.
    word = "".join(dataset.id_to_char[i] for i in x)
    plot_word_trajectory(word, item["y"], title_prefix=title_prefix)


def plot_prediction(
    word: str,
    target: Sequence[float],
    prediction: Sequence[float],
    title_prefix: str = "Prediction vs Target",
) -> None:
    # Plot predicted and target trajectories together for comparison.
    y_true = list(target)
    y_pred = list(prediction)
    x_true = list(range(len(y_true)))
    x_pred = list(range(len(y_pred)))
    plt.figure(figsize=(8, 3))
    plt.plot(x_true, y_true, label="target", linewidth=1.5)
    plt.plot(x_pred, y_pred, label="prediction", linewidth=1.5, linestyle="--")
    plt.title(f"{title_prefix}: {word}")
    plt.xlabel("Time index")
    plt.ylabel("Trajectory value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_loss_plot(history: dict, path: str) -> None:
    # Save a loss curve plot to disk.
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(6, 3))
    if train_loss:
        plt.plot(epochs, train_loss, label="train", linewidth=1.5)
    if val_loss:
        plt.plot(epochs, val_loss, label="val", linewidth=1.5)
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
