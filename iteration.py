from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hyper_params import HyperParams
from dataset import SourGrapeDataset, PhonemeDataset
from model import LSTMRegressor, Seq2SeqRegressor, PhonemeRegressor
from train_eval import eval_last_epoch, eval_one_epoch, train_one_epoch
from utils import (
    save_loss_plot,
    save_prediction_plot,
    save_mean_trajectory_drift,
    save_loss_drift,
    save_embedding_pca,
)
from datetime import datetime


def run_phoneme_pretrain(
    hp: HyperParams,
    seed: int,
    phoneme_dataset: PhonemeDataset,
    device: torch.device,
    out_dir: Path,
) -> torch.Tensor:
    # Reproducibility for the pretraining stage.
    torch.manual_seed(seed)

    # Output directory for pretraining artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "pretrain_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Dataset and dataloaders (configurable split).
    pretrain_dataset = phoneme_dataset
    train_ds, test_ds = torch.utils.data.random_split(
        pretrain_dataset,
        hp.data_split_ratio,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)

    # Model initialization.
    vocab = phoneme_dataset.build_vocab().char_to_id
    model = PhonemeRegressor(len(vocab), embed_size=hp.embed_size).to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.pretrain_lr)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
    }
    for epoch in range(1, hp.pretrain_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, training_type="pretrain"
        )
        test_loss = eval_one_epoch(
            model, test_loader, device, loss_fn, training_type="pretrain"
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(
            f"pretrain_epoch={epoch}, train_loss={train_loss:.6f}, "
            f"test_loss={test_loss:.6f}"
        )
        # Save a checkpoint every five epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"pretrain_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save PCA plot of embedding weights from the pretraining stage.
    emb = model.embedding.weight.detach().cpu().numpy()
    pca_path = out_dir / "embedding_pca.png"
    save_embedding_pca(emb, phoneme_dataset.build_vocab().id_to_char, str(pca_path))

    # Save a single loss plot for the pretraining stage.
    loss_plot_path = out_dir / "pretrain_loss_curve.png"
    save_loss_plot(history, str(loss_plot_path))

    # Save pretraining history as CSV.
    history_path = out_dir / "pretrain_history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,subset,loss\n")
        for epoch, loss in enumerate(history["train_loss"], start=1):
            f.write(f"{epoch},train,{loss}\n")
        for epoch, loss in enumerate(history["test_loss"], start=1):
            f.write(f"{epoch},test,{loss}\n")

    return model.embedding.weight.detach().cpu()


def run_trajectory_training(
    hp: HyperParams,
    seed: int,
    gen: int,
    trajectory_train_dataset: SourGrapeDataset,
    trajectory_test_dataset: SourGrapeDataset,
    model_type: str,
    embedding_weights: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    resume_path: str = "",
) -> None:
    # Reproducibility.
    torch.manual_seed(seed)

    # Dataset and dataloaders.
    ds_a, ds_b, ds_c = trajectory_train_dataset.split_dataset(seed=seed)

    # Rotate train split by generation number.
    mod = gen % 3
    if mod == 0:
        train_ds = torch.utils.data.ConcatDataset([ds_a, ds_b])
    elif mod == 1:
        train_ds = torch.utils.data.ConcatDataset([ds_b, ds_c])
    else:
        train_ds = torch.utils.data.ConcatDataset([ds_c, ds_a])

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(
        trajectory_test_dataset, batch_size=hp.batch_size, shuffle=False
    )

    # Model selection.
    if model_type == "seq2seq":
        model = Seq2SeqRegressor(
            input_size=len(trajectory_train_dataset.vocab.char_to_id),
            output_len=trajectory_train_dataset.trajectory_len,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    else:
        model = LSTMRegressor(
            input_size=len(trajectory_train_dataset.vocab.char_to_id),
            output_size=trajectory_train_dataset.trajectory_len,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = (targets != hp.trajectory_pad_value).to(targets.dtype)
        return F.mse_loss(preds, targets, reduction="sum", weight=weight)

    # Optionally resume from a checkpoint.
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Output directory for artifacts.
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
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, training_type="train"
        )
        test_loss = eval_one_epoch(
            model, test_loader, device, loss_fn, training_type="train"
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history_rows.append((epoch, "train", train_loss))
        history_rows.append((epoch, "test", test_loss))
        print(f"epoch={epoch}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        # Save a checkpoint every five epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save a single loss plot for this run.
    plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(plot_path))

    # Final evaluation and prediction capture for next generation.
    final_loss, preds = eval_last_epoch(
        model, test_loader, device, loss_fn, training_type="train"
    )
    history["final_test_loss"] = final_loss
    history_rows.append((hp.epochs, "final_test", final_loss))
    print(f"final_test_loss={final_loss:.6f}")

    mask_before = trajectory_train_dataset.y_prev != hp.trajectory_pad_value
    prev_before = trajectory_train_dataset.y_prev[mask_before].mean().item()
    # Map predictions to training items by word.
    pred_map = {}
    for i, word in enumerate(trajectory_test_dataset.words):
        if word not in pred_map:
            pred_map[word] = preds[i]
    preds_train = []
    for word in trajectory_train_dataset.words:
        if word not in pred_map:
            raise ValueError(f"Missing prediction for word: {word}")
        preds_train.append(pred_map[word])
    preds_train = torch.stack(preds_train, dim=0)
    trajectory_train_dataset.update_prev_targets(preds_train)
    mask_after = trajectory_train_dataset.y_prev != hp.trajectory_pad_value
    prev_after = trajectory_train_dataset.y_prev[mask_after].mean().item()
    print(f"y_prev_mean_before={prev_before:.6f}, y_prev_mean_after={prev_after:.6f}")

    # Save one prediction plot per item type (ignore padded values in visualization).
    seen_types = set()
    for idx in range(len(trajectory_test_dataset)):
        item_type = trajectory_test_dataset[idx]["item_type"]
        if item_type in seen_types:
            continue
        word = "".join(
            trajectory_test_dataset.vocab.id_to_char[i]
            for i in trajectory_test_dataset[idx]["x"].tolist()
        )
        target = trajectory_test_dataset[idx]["y_real"].tolist()
        prediction = preds[idx].tolist()
        pad_value = hp.trajectory_pad_value
        mask = [t != pad_value for t in target]
        target = [t for t, keep in zip(target, mask) if keep]
        prediction = [p for p, keep in zip(prediction, mask) if keep]
        pred_path = out_dir / f"prediction_vs_target_{item_type}.png"
        save_prediction_plot(word, target, prediction, str(pred_path))
        seen_types.add(item_type)
        if len(seen_types) >= 5:
            break

    # Save training history as CSV.
    history_path = out_dir / "history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,subset,loss\n")
        for epoch, subset, loss in history_rows:
            f.write(f"{epoch},{subset},{loss}\n")
    np.save(out_dir / "predictions.npy", preds.numpy())


def run_generations(condition: str, num_generations: int) -> None:
    # Train multiple generations with different random seeds.
    hp = HyperParams()
    device = torch.device(hp.device)
    
    # Phoneme dataset for the pretraining stage.
    phoneme_dataset = PhonemeDataset(
        condition=condition,
        data_path=hp.phoneme_data_path,
        augment=True,
    )
    vocab = phoneme_dataset.build_vocab()
    
    # Trajectory datasets: training uses augmentation, testing uses raw targets.
    trajectory_train_dataset = SourGrapeDataset(
        condition=condition,
        data_path=hp.train_data_path,
        npy_root=hp.npy_root,
        augment=True,
        vocab=vocab,
    )
    trajectory_test_dataset = SourGrapeDataset(
        condition=condition,
        data_path=hp.test_data_path,
        npy_root=hp.npy_root,
        augment=False,
        vocab=vocab,
    )
    
    # Collect predictions across generations for visualization.
    preds_by_gen = {}
    
    # Output folder for all generations in this run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(hp.output_root) / f"{condition}_{timestamp}"
    
    # Each generation runs two named stages in sequence.
    stage_names = ["phoneme_pretrain", "trajectory_training"]
    for gen in range(0, num_generations):
        # Log the planned stages for readability.
        print(f"gen={gen}, stages={stage_names}")
        gen_out_dir = run_root / f"gen_{gen}"
        
        # Stage 1: phoneme pretrain (initialize stage-specific inputs first).
        embedding_weights = run_phoneme_pretrain(
            phoneme_dataset=phoneme_dataset,
            hp=hp,
            seed=hp.seed + gen,
            device=device,
            out_dir=gen_out_dir,
        )
        
        # Stage 2: trajectory training (initialize stage-specific inputs first).
        run_trajectory_training(
            hp=hp,
            seed=hp.seed + gen,
            gen=gen,
            trajectory_train_dataset=trajectory_train_dataset,
            trajectory_test_dataset=trajectory_test_dataset,
            model_type=hp.model_type,
            embedding_weights=embedding_weights,
            device=device,
            out_dir=gen_out_dir,
        )
        pred_path = run_root / f"gen_{gen}" / "predictions.npy"
        if pred_path.exists():
            preds_by_gen[gen] = np.load(pred_path)

    # Save mean trajectory drift plots across generations (ignore padded values).
    drift_dir = run_root / "drift_plots"
    drift_dir.mkdir(parents=True, exist_ok=True)
    item_types = list(trajectory_test_dataset.item_types)
    unique_types = sorted(set(item_types))
    targets = trajectory_test_dataset.y_real.numpy()
    for idx_type, item_type in enumerate(unique_types):
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        mean_by_gen = {}
        targets_subset = targets[idxs]
        mask = targets_subset != hp.trajectory_pad_value
        for gen, preds in preds_by_gen.items():
            preds_subset = preds[idxs]
            masked = np.where(mask, preds_subset, np.nan)
            mean_by_gen[gen] = np.nanmean(masked, axis=0)
        mean_by_gen["target"] = np.nanmean(
            np.where(mask, targets_subset, np.nan), axis=0
        )
        safe_type = "".join(ch for ch in str(item_type) if ch.isalnum() or ch in "_-")
        if not safe_type:
            safe_type = f"type_{idx_type}"
        out_path = drift_dir / f"mean_drift_{safe_type}.png"
        save_mean_trajectory_drift(mean_by_gen, str(out_path))

    # Save loss drift plot across generations.
    history_by_gen = {}
    for gen in range(0, num_generations):
        history_path = run_root / f"gen_{gen}" / "history.csv"
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
    run_generations(num_generations=5, condition="glide")
    run_generations(num_generations=5, condition="fricative")
