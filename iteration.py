from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import hyper_params as hp
from dataset import RepeatShuffleSampler, SourGrapeDataset, PhonemeDataset
from model import LSTMRegressor, Seq2SeqRegressor, PhonemeRegressor
from train_eval import eval_last_epoch, eval_one_epoch, train_one_epoch
from utils import (
    save_loss_plot,
    save_prediction_plot,
    save_trajectory_drift,
    save_loss_drift,
    save_embedding_plot,
)
from datetime import datetime


def run_phoneme_pretrain(
    seed: int,
    phoneme_dataset: PhonemeDataset,
    device: torch.device,
    out_dir: Path,
) -> tuple[torch.Tensor, list[tuple[int, str, float]]]:
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
    history_rows = []
    for epoch in range(1, hp.pretrain_epochs + 1):
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, training_type="pretrain"
        )
        test_loss, _, _ = eval_one_epoch(
            model, test_loader, device, loss_fn, training_type="pretrain"
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history_rows.append((epoch, "train", train_loss))
        history_rows.append((epoch, "test", test_loss))
        print(
            f"pretrain_epoch={epoch}, train_loss={train_loss:.6f}, "
            f"test_loss={test_loss:.6f}"
        )
        # Save a checkpoint every five epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"pretrain_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save the pretraining embedding plot.
    emb = model.embedding.weight.detach().cpu().numpy()
    embedding_path = out_dir / "pretrain_embedding.png"
    save_embedding_plot(emb, phoneme_dataset.vocab.id_to_char, str(embedding_path))

    # Save a single loss plot for the pretraining stage.
    loss_plot_path = out_dir / "pretrain_loss_curve.png"
    save_loss_plot(history, str(loss_plot_path))

    return model.embedding.weight.detach().cpu(), history_rows


def run_trajectory_training(
    seed: int,
    trajectory_dataset: SourGrapeDataset,
    embedding_weights: torch.Tensor | None,
    device: torch.device,
    out_dir: Path,
    resume_path: str = "",
) -> tuple[np.ndarray, list[tuple[int, str, float, float, float]]]:
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
    if hp.model_type == "seq2seq":
        model = Seq2SeqRegressor(
            input_size=len(trajectory_dataset.vocab.char_to_id),
            output_len=trajectory_dataset.max_trajectory_len,
            bidirectional=hp.bidirectional,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    else:
        model = LSTMRegressor(
            input_size=len(trajectory_dataset.vocab.char_to_id),
            output_size=trajectory_dataset.max_trajectory_len,
            bidirectional=hp.bidirectional,
            embedding_weights=embedding_weights,
            freeze_embedding=embedding_weights is not None,
        )
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ignore padded trajectory positions.
        mask = targets != hp.padding_value
        return F.mse_loss(preds[mask], targets[mask], reduction="sum")

    def penalty_loss_fn(
        preds: torch.Tensor,
        penalty_targets: torch.Tensor,
    ) -> torch.Tensor:
        # Keep only positive penalty targets.
        mask = penalty_targets == 1
        if hp.penalty_loss_type == "sigmoid_bce":
            pred_activity = torch.sigmoid(
                hp.penalty_scale * (preds - hp.penalty_threshold)
            )
            return F.binary_cross_entropy(
                pred_activity[mask],
                penalty_targets[mask],
                reduction="sum",
            )
        if hp.penalty_loss_type == "relu_mse":
            pred_activity = torch.relu(
                hp.penalty_scale * (preds - hp.penalty_threshold)
            )
            return F.mse_loss(
                pred_activity[mask],
                penalty_targets[mask],
                reduction="sum",
            )
        if hp.penalty_loss_type == "softplus_mse":
            pred_activity = F.softplus(
                hp.penalty_scale * (preds - hp.penalty_threshold)
            )
            return F.mse_loss(
                pred_activity[mask],
                penalty_targets[mask],
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
        pred_path = out_dir / f"prediction_{item_type}.png"
        save_prediction_plot(word, target, prediction, str(pred_path))
        seen_types.add(item_type)
        if len(seen_types) >= 5:
            break

    return preds.numpy(), history_rows


def run_generations(
    iteration_root: Path,
    seed: int = hp.seed,
    condition: str = "glide",
    num_generations: int = hp.generations,
    stage: str = hp.stage,
    device: torch.device = torch.device(hp.device),
) -> None:
    # Run one full generation chain for a single condition.
    iteration_seed = seed
    
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
    pretrain_history_by_gen = {}
    train_history_by_gen = {}
    
    for gen in range(0, num_generations):
        generation_seed = iteration_seed + gen
        print(f"gen={gen}, stage={stage}, seed={generation_seed}")
        gen_out_dir = iteration_root / f"{condition}_gen_{gen}"
        embedding_weights = None

        if stage in {"all", "pretrain"}:
            # Run phoneme pretraining for this generation.
            embedding_weights, pretrain_history_by_gen[gen] = run_phoneme_pretrain(
                seed=generation_seed,
                phoneme_dataset=phoneme_dataset,
                device=device,
                out_dir=gen_out_dir,
            )

        if stage in {"all", "train"}:
            # Run trajectory training for this generation.
            preds_by_gen[gen], train_history_by_gen[gen] = run_trajectory_training(
                seed=generation_seed,
                trajectory_dataset=trajectory_dataset,
                embedding_weights=embedding_weights,
                device=device,
                out_dir=gen_out_dir,
            )

    if not preds_by_gen:
        return

    # Save the condition summary artifacts.
    summary_dir = iteration_root / f"{condition}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    item_types = list(trajectory_dataset.item_types)
    unique_types = sorted(set(item_types))
    words = trajectory_dataset.words
    targets = trajectory_dataset.pad_targets(trajectory_dataset.y_real).numpy()
    
    # Save the trajectory drift plots.
    for idx_type, item_type in enumerate(unique_types):
        idxs = [i for i, t in enumerate(item_types) if t == item_type]
        if not idxs:
            continue
        stats_by_gen = {}
        targets_subset = targets[idxs]
        mask = targets_subset != hp.padding_value
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
        trajectory_drift_path = summary_dir / f"prediction_drift_{safe_type}.png"
        save_trajectory_drift(stats_by_gen, str(trajectory_drift_path))

    # Save all generation predictions in one CSV file.
    preds_csv_path = summary_dir / "predictions.csv"
    with open(preds_csv_path, "w", encoding="utf-8") as f:
        max_len = max(len(target) for target in trajectory_dataset.y_real)
        timestep_cols = ",".join(f"timestep_{idx}" for idx in range(max_len))
        f.write(f"generation,item_index,word,item_type,{timestep_cols}\n")
        for gen, preds in preds_by_gen.items():
            for idx, pred in enumerate(preds):
                valid_pred = pred[: len(trajectory_dataset.y_real[idx])]
                padded_pred = list(valid_pred) + [""] * (max_len - len(valid_pred))
                pred_values = ",".join(str(value) for value in padded_pred)
                f.write(f"{gen},{idx},{words[idx]},{item_types[idx]},{pred_values}\n")

    # Save the combined pretraining history.
    if pretrain_history_by_gen:
        pretrain_history_path = summary_dir / "pretrain_history.csv"
        with open(pretrain_history_path, "w", encoding="utf-8") as f:
            f.write("generation,epoch,subset,loss\n")
            for gen, rows in pretrain_history_by_gen.items():
                for epoch, subset, loss in rows:
                    f.write(f"{gen},{epoch},{subset},{loss}\n")

    # Save the combined trajectory history.
    if train_history_by_gen:
        history_path = summary_dir / "history.csv"
        with open(history_path, "w", encoding="utf-8") as f:
            f.write("generation,epoch,subset,loss,main_loss,penalty_loss\n")
            for gen, rows in train_history_by_gen.items():
                for epoch, subset, loss, main_loss, penalty_loss in rows:
                    f.write(
                        f"{gen},{epoch},{subset},{loss},{main_loss},{penalty_loss}\n"
                    )

    # Save loss drift plot across generations.
    history_by_gen = {}
    for gen, rows in train_history_by_gen.items():
        train_loss = []
        test_loss = []
        for _, subset, loss, _, _ in rows:
            if subset == "train":
                train_loss.append(float(loss))
            elif subset == "test":
                test_loss.append(float(loss))
        history_by_gen[gen] = {"train_loss": train_loss, "test_loss": test_loss}
    loss_drift_path = summary_dir / "loss_drift.png"
    save_loss_drift(history_by_gen, str(loss_drift_path))


def run_iterations(
    seed: int = hp.seed,
    num_iterations: int = hp.iterations,
    num_generations: int = hp.generations,
    stage: str = hp.stage,
) -> None:
    # Select the device for this call.
    if hp.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(hp.device)

    # Create the output folder for this call.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(hp.output_root) / f"iterations_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Save the run arguments and hyperparameters.
    run_config_path = run_root / "run_config.txt"
    with open(run_config_path, "w", encoding="utf-8") as f:
        f.write("[parsed_args]\n")
        f.write(f"seed = {seed}\n")
        f.write(f"num_iterations = {num_iterations}\n")
        f.write(f"num_generations = {num_generations}\n")
        f.write(f"stage = {stage}\n\n")
        f.write("[hyperparameters]\n")
        for key, value in vars(hp).items():
            if key.startswith("_") or callable(value):
                continue
            f.write(f"{key} = {value}\n")

    # Run all iterations.
    for iteration in range(num_iterations):
        iteration_seed = seed + iteration * num_generations
        print(f"iteration={iteration}, seed={iteration_seed}")
        iteration_root = run_root / f"iteration_{iteration}"
        iteration_root.mkdir(parents=True, exist_ok=True)

        # Run all conditions for this iteration.
        for condition in hp.conditions:
            run_generations(
                iteration_root=iteration_root,
                seed=iteration_seed,
                condition=condition,
                num_generations=num_generations,
                stage=stage,
                device=device,
            )
