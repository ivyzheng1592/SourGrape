import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hyper_params import HyperParams
from dataset import SourGrapeDataset
from model import LSTMRegressor, Seq2SeqRegressor
from train_eval import eval_last_epoch, eval_one_epoch, train_one_epoch
from utils import (
    save_loss_plot,
    save_prediction_plot,
    save_mean_trajectory_drift,
    save_loss_drift,
    save_embedding_pca,
)


def build_model(dataset: SourGrapeDataset, model_type: str) -> nn.Module:
    # Build model for this generation.
    if model_type == "seq2seq":
        return Seq2SeqRegressor(
            input_size=len(dataset.vocab),
            output_len=dataset.trajectory_len,
        )
    return LSTMRegressor(
        input_size=len(dataset.vocab),
        output_size=dataset.trajectory_len,
    )


def masked_mse(preds: torch.Tensor, targets: torch.Tensor, pad_value: float) -> torch.Tensor:
    # Mask padded regions in loss.
    weight = (targets != pad_value).to(preds.dtype)
    diff = (preds - targets) ** 2
    return (diff * weight).sum()


def sync_dataset_lengths(dataset: SourGrapeDataset, max_len: int, pad_value: float) -> None:
    # Pad dataset targets to match max_len.
    if max_len <= dataset.trajectory_len:
        return
    pad_len = max_len - dataset.trajectory_len
    pad = torch.full(
        (dataset.y_real.shape[0], pad_len),
        pad_value,
        dtype=dataset.y_real.dtype,
    )
    dataset.y_real = torch.cat([dataset.y_real, pad], dim=1)
    dataset.y_prev = torch.cat([dataset.y_prev, pad.clone()], dim=1)
    dataset.trajectory_len = max_len


def iterate_once(
    hp: HyperParams,
    condition: str,
    generation: int,
    seed: int,
    train_dataset: SourGrapeDataset,
    test_dataset: SourGrapeDataset,
    model_type: str,
    device: torch.device,
    resume_path: str = "",
) -> None:

    # Reproducibility.
    torch.manual_seed(seed)

    # Dataset and dataloaders.
    ds_a, ds_b, ds_c = train_dataset.split_dataset(seed=seed)

    # Rotate train split by generation number.
    mod = generation % 3
    if mod == 0:
        train_ds = ds_a
    elif mod == 1:
        train_ds = ds_b
    else:
        train_ds = ds_c

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

    # Model selection.
    model = build_model(train_dataset, model_type)
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = lambda preds, targets: masked_mse(preds, targets, hp.trajectory_pad_value)

    # Optionally resume from a checkpoint.
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Output directory for artifacts.
    out_dir = Path(hp.output_root) / condition / f"gen_{generation}"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
    }
    history_rows = []
    for epoch in range(1, hp.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        test_loss = eval_one_epoch(model, test_loader, device, loss_fn)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history_rows.append((epoch, "train", train_loss))
        history_rows.append((epoch, "test", test_loss))
        print(f"epoch={epoch}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        # Save a checkpoint every epoch.
        ckpt_path = model_dir / f"model_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)

    # Save a single loss plot for this run.
    plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(plot_path))

    # Final evaluation and prediction capture for next generation.
    final_loss, preds = eval_last_epoch(model, test_loader, device, loss_fn)
    history["final_test_loss"] = final_loss
    history_rows.append((hp.epochs, "final_test", final_loss))
    print(f"final_test_loss={final_loss:.6f}")
    prev_before = train_dataset.y_prev.mean().item()
    # If max length grew, pad y_real/y_prev to match and update trajectory_len.
    sync_dataset_lengths(train_dataset, preds.shape[1], hp.trajectory_pad_value)
    # Map predictions to training items by word.
    pred_map = {}
    for i, word in enumerate(test_dataset.words):
        if word not in pred_map:
            pred_map[word] = preds[i]
    preds_train = []
    for word in train_dataset.words:
        if word not in pred_map:
            raise ValueError(f"Missing prediction for word: {word}")
        preds_train.append(pred_map[word])
    preds_train = torch.stack(preds_train, dim=0)
    train_dataset.update_prev_targets(preds_train)
    prev_after = train_dataset.y_prev.mean().item()
    print(f"y_prev_mean_before={prev_before:.6f}, y_prev_mean_after={prev_after:.6f}")

    # Save one prediction plot per item type.
    seen_types = set()
    for idx in range(len(test_dataset)):
        item_type = test_dataset[idx]["item_type"]
        if item_type in seen_types:
            continue
        word = "".join(test_dataset.id_to_char[i] for i in test_dataset[idx]["x"].tolist())
        target = test_dataset[idx]["y_real"].tolist()
        prediction = preds[idx].tolist()
        pred_path = out_dir / f"prediction_vs_target_{item_type}.png"
        save_prediction_plot(word, target, prediction, str(pred_path))
        seen_types.add(item_type)
        if len(seen_types) >= 5:
            break

    # Save PCA plot of embedding weights.
    if hasattr(model, "embedding"):
        emb = model.embedding.weight.detach().cpu().numpy()
        pca_path = out_dir / "embedding_pca.png"
        save_embedding_pca(emb, train_dataset.id_to_char, str(pca_path))

    # Save artifacts.
    torch.save(model.state_dict(), model_dir / "model.pt")
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.vocab, f, ensure_ascii=True, indent=2)
    history_path = out_dir / "history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,subset,loss\n")
        for epoch, subset, loss in history_rows:
            f.write(f"{epoch},{subset},{loss}\n")
    np.save(out_dir / "predictions.npy", preds.numpy())


def iterate_multi(condition: str, num_generations: int) -> None:
    # Train multiple generations with different random seeds.
    hp = HyperParams()
    train_dataset = SourGrapeDataset(
        condition=condition,
        data_path=hp.train_data_path,
        augment=True,
    )
    test_dataset = SourGrapeDataset(
        condition=condition,
        data_path=hp.test_data_path,
        augment=False,
    )
    model_type = hp.model_type
    device = torch.device(hp.device)
    preds_by_gen = {}
    for gen in range(0, num_generations):
        iterate_once(
            hp=hp,
            condition=condition,
            generation=gen,
            seed=hp.seed + gen,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_type=model_type,
            device=device,
        )
        pred_path = Path(hp.output_root) / condition / f"gen_{gen}" / "predictions.npy"
        if pred_path.exists():
            preds_by_gen[gen] = np.load(pred_path)

    # Save mean trajectory drift plots across generations.
    drift_dir = Path(hp.output_root) / condition / "drift_plots"
    drift_dir.mkdir(parents=True, exist_ok=True)
    item_types = list(test_dataset.item_types)
    unique_types = sorted(set(item_types))
    targets = test_dataset.y_real.numpy()
    for item_type in unique_types:
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        mean_by_gen = {}
        for gen, preds in preds_by_gen.items():
            mean_by_gen[gen] = preds[idxs].mean(axis=0)
        mean_by_gen["target"] = targets[idxs].mean(axis=0)
        out_path = drift_dir / f"mean_drift_{item_type}.png"
    save_mean_trajectory_drift(mean_by_gen, str(out_path))

    # Save loss drift plot across generations.
    history_by_gen = {}
    for gen in range(0, num_generations):
        history_path = Path(hp.output_root) / condition / f"gen_{gen}" / "history.csv"
        if not history_path.exists():
            continue
        train_loss = []
        test_loss = []
        with open(history_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                epoch_str, subset, loss_str = line.strip().split(",")
                if subset == "train":
                    train_loss.append(float(loss_str))
                elif subset == "test":
                    test_loss.append(float(loss_str))
        history_by_gen[gen] = {"train_loss": train_loss, "test_loss": test_loss}

    loss_drift_path = drift_dir / "loss_drift.png"
    save_loss_drift(history_by_gen, str(loss_drift_path))


if __name__ == "__main__":
    iterate_multi(num_generations=5, condition="glide")
    iterate_multi(num_generations=5, condition="fricative")
