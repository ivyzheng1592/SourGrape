from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hyper_params import HyperParams
from dataset import RepeatShuffleSampler, SourGrapeDataset, PhonemeDataset
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
    phoneme_dataset.vocab.save(out_dir / "vocab.json")

    # Build the training and test loaders.
    train_ds, test_ds = torch.utils.data.random_split(
        phoneme_dataset,
        hp.pretrain_data_split_ratio,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)

    # Model initialization.
    vocab = phoneme_dataset.vocab.char_to_id
    model = PhonemeRegressor(
        vocab_size=len(vocab), 
        embed_size=hp.embed_size
    ).to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.pretrain_lr)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
    }
    for epoch in range(1, hp.pretrain_epochs + 1):
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, training_type="pretrain"
        )
        test_loss, _, _ = eval_one_epoch(
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
    save_embedding_pca(emb, phoneme_dataset.vocab.id_to_char, str(pca_path))

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
    trajectory_dataset: SourGrapeDataset,
    model_type: str,
    embedding_weights: torch.Tensor | None,
    device: torch.device,
    out_dir: Path,
    resume_path: str = "",
) -> list[torch.Tensor]:
    # Reproducibility.
    torch.manual_seed(seed)

    # Output directory for artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build the training and test loaders.
    train_sampler = RepeatShuffleSampler(
        dataset_size=len(trajectory_dataset),
        repeats=hp.train_repeats_per_epoch,
        seed=seed,
    )
    train_loader = DataLoader(
        trajectory_dataset,
        batch_size=hp.batch_size,
        sampler=train_sampler,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=True),
    )
    test_loader = DataLoader(
        trajectory_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=False),
    )

    # Model selection.
    if model_type == "seq2seq":
        model = Seq2SeqRegressor(
            input_size=len(trajectory_dataset.vocab.char_to_id),
            output_len=trajectory_dataset.max_trajectory_len,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    else:
        model = LSTMRegressor(
            input_size=len(trajectory_dataset.vocab.char_to_id),
            output_size=trajectory_dataset.max_trajectory_len,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = (targets != hp.trajectory_pad_value).to(targets.dtype)
        return F.mse_loss(preds, targets, reduction="sum", weight=weight)

    def penalty_loss_fn(
        preds: torch.Tensor,
        penalty_targets: torch.Tensor,
    ) -> torch.Tensor:
        pred_activity = torch.sigmoid(
            hp.penalty_sigmoid_scale * (preds - hp.penalty_threshold)
        )
        weight = penalty_targets != hp.trajectory_pad_value
        return F.binary_cross_entropy(
            pred_activity[weight],
            penalty_targets[weight],
            reduction="sum",
        )

    # Optionally resume from a checkpoint.
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_main_loss": [],
        "test_main_loss": [],
        "train_penalty_loss": [],
        "test_penalty_loss": [],
    }
    history_rows = []
    for epoch in range(1, hp.epochs + 1):
        train_loss, train_main_loss, train_penalty_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn,
            aux_loss_fn=penalty_loss_fn,
            aux_loss_weight=hp.penalty_loss_weight,
            training_type="train",
        )
        test_loss, test_main_loss, test_penalty_loss = eval_one_epoch(
            model,
            test_loader,
            device,
            loss_fn,
            aux_loss_fn=penalty_loss_fn,
            aux_loss_weight=hp.penalty_loss_weight,
            training_type="train",
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_main_loss"].append(train_main_loss)
        history["test_main_loss"].append(test_main_loss)
        history["train_penalty_loss"].append(train_penalty_loss)
        history["test_penalty_loss"].append(test_penalty_loss)
        history_rows.append((epoch, "train", train_loss, train_main_loss, train_penalty_loss))
        history_rows.append((epoch, "test", test_loss, test_main_loss, test_penalty_loss))
        print(
            f"epoch={epoch}, "
            f"train_loss={train_loss:.6f}, train_main_loss={train_main_loss:.6f}, "
            f"train_penalty_loss={train_penalty_loss:.6f}, "
            f"test_loss={test_loss:.6f}, test_main_loss={test_main_loss:.6f}, "
            f"test_penalty_loss={test_penalty_loss:.6f}"
        )
        # Save a checkpoint every five epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save a single loss plot for this run.
    loss_plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(loss_plot_path))

    # Evaluate the model on the full trajectory dataset.
    final_loss, final_main_loss, final_penalty_loss, preds = eval_last_epoch(
        model,
        test_loader,
        device,
        loss_fn,
        aux_loss_fn=penalty_loss_fn,
        aux_loss_weight=hp.penalty_loss_weight,
        training_type="train",
    )
    history["final_test_loss"] = final_loss
    history["final_test_main_loss"] = final_main_loss
    history["final_test_penalty_loss"] = final_penalty_loss
    history_rows.append((hp.epochs, "final_test", final_loss, final_main_loss, final_penalty_loss))
    print(
        f"final_test_loss={final_loss:.6f}, "
        f"final_test_main_loss={final_main_loss:.6f}, "
        f"final_test_penalty_loss={final_penalty_loss:.6f}"
    )

    # Get the mean value of y_prev before updating it.
    masked_mean_before = torch.cat(trajectory_dataset.y_prev).mean().item()
    # Update y_prev with the current generation predictions.
    trajectory_dataset.update_prev_targets(preds)
    # Get the mean value of y_prev after updating it.
    masked_mean_after = torch.cat(trajectory_dataset.y_prev).mean().item()
    print(
        f"y_prev_mean_before={masked_mean_before:.6f}, "
        f"y_prev_mean_after={masked_mean_after:.6f}"
    )

    # Save one prediction plot per item type.
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
        pred_path = out_dir / f"prediction_vs_target_{item_type}.png"
        save_prediction_plot(word, target, prediction, str(pred_path))
        seen_types.add(item_type)
        if len(seen_types) >= 5:
            break

    # Save training history as CSV.
    history_path = out_dir / "history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,subset,loss,main_loss,penalty_loss\n")
        for epoch, subset, loss, main_loss, penalty_loss in history_rows:
            f.write(f"{epoch},{subset},{loss},{main_loss},{penalty_loss}\n")
    np.save(out_dir / "predictions.npy", preds.numpy())
    
    return preds.numpy()


def run_generations(
    seed: int | None = None,
    condition: str = "glide",
    num_generations: int = 5,
    stage: str = "all",
) -> None:
    # Train multiple generations with different random seeds.
    hp = HyperParams()
    if seed is not None:
        hp.seed = seed

    # Select device.
    if hp.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(hp.device)
    
    # Load the phoneme dataset for the pretraining stage.
    phoneme_dataset = PhonemeDataset(
        condition=condition,
        data_path=hp.phoneme_data_path,
        augment=True,
    )
    vocab = phoneme_dataset.vocab
    
    # Load the trajectory dataset for the training stage.
    trajectory_dataset = SourGrapeDataset(
        vocab=vocab,
        condition=condition,
        trajectory_data_path=hp.trajectory_data_path,
        trajectory_npy_root=hp.trajectory_npy_root,
        penalty_data_path=hp.penalty_data_path,
        penalty_npy_root=hp.penalty_npy_root,
    )
    
    # Store the predictions from each generation.
    preds_by_gen = {}
    
    # Output folder for all generations in this run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(hp.output_root) / f"{condition}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Save the run arguments and hyperparameters.
    run_config_path = run_root / "run_config.txt"
    with open(run_config_path, "w", encoding="utf-8") as f:
        f.write("[parsed_args]\n")
        f.write(f"seed = {seed}\n")
        f.write(f"condition = {condition}\n")
        f.write(f"num_generations = {num_generations}\n")
        f.write(f"stage = {stage}\n\n")
        f.write("[hyperparameters]\n")
        for key, value in asdict(hp).items():
            f.write(f"{key} = {value}\n")
    
    for gen in range(0, num_generations):
        print("gen=%d, stage=%s" % (gen, stage))
        gen_out_dir = run_root / f"gen_{gen}"
        embedding_weights = None

        if stage in {"all", "pretrain"}:
            # Run phoneme pretraining for this generation.
            embedding_weights = run_phoneme_pretrain(
                hp=hp,
                seed=hp.seed + gen,
                phoneme_dataset=phoneme_dataset,
                device=device,
                out_dir=gen_out_dir,
            )

        if stage in {"all", "train"}:
            # Run trajectory training for this generation.
            preds_by_gen[gen] = run_trajectory_training(
                hp=hp,
                seed=hp.seed + gen,
                trajectory_dataset=trajectory_dataset,
                model_type=hp.model_type,
                embedding_weights=embedding_weights,
                device=device,
                out_dir=gen_out_dir,
            )

    if not preds_by_gen:
        return

    # Save the mean trajectory drift plots.
    drift_dir = run_root / "drift_plots"
    drift_dir.mkdir(parents=True, exist_ok=True)
    item_types = list(trajectory_dataset.item_types)
    unique_types = sorted(set(item_types))
    targets = trajectory_dataset.pad_targets(trajectory_dataset.y_real).numpy()
    for idx_type, item_type in enumerate(unique_types):
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        stats_by_gen = {}
        targets_subset = targets[idxs]
        mask = targets_subset != hp.trajectory_pad_value
        for gen, preds in preds_by_gen.items():
            preds_subset = preds[idxs]
            masked = np.where(mask, preds_subset, np.nan)
            stats_by_gen[gen] = {
                "mean": np.nanmean(masked, axis=0),
                "std": np.nanstd(masked, axis=0),
            }
        masked_targets = np.where(mask, targets_subset, np.nan)
        stats_by_gen["target"] = {
            "mean": np.nanmean(masked_targets, axis=0),
            "std": np.nanstd(masked_targets, axis=0),
        }
        safe_type = "".join(ch for ch in str(item_type) if ch.isalnum() or ch in "_-")
        if not safe_type:
            safe_type = f"type_{idx_type}"
        out_path = drift_dir / f"mean_drift_{safe_type}.png"
        save_mean_trajectory_drift(stats_by_gen, str(out_path))

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
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                _, subset, loss_str = parts[:3]
                if subset == "train":
                    train_loss.append(float(loss_str))
                elif subset == "test":
                    test_loss.append(float(loss_str))
        history_by_gen[gen] = {"train_loss": train_loss, "test_loss": test_loss}
    loss_drift_path = drift_dir / "loss_drift.png"
    save_loss_drift(history_by_gen, str(loss_drift_path))
