from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import hyper_params as hp
from dataset import RepeatShuffleSampler, SourGrapeDataset, PhonemeDataset
from model import LSTMRegressor, Seq2SeqRegressor, PhonemeRegressor
from train_eval import eval_last_epoch, eval_one_epoch, train_one_epoch
from utils import (
    save_embedding_plot,
    save_history_csv,
    save_loss_drift_plot,
    save_loss_plot,
    save_predictions_csv,
    save_trajectory_plots,
    save_trajectory_drift_plots,
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
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, stage="pretrain", device=device
        )
        test_loss = eval_one_epoch(
            model, test_loader, loss_fn, stage="pretrain", device=device
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


def get_generation_labels(generation: int) -> tuple[set[str], str]:
    # Return the exposed subset labels and held-out subset label.
    if generation % 3 == 0:
        return {"a", "b"}, "c"
    if generation % 3 == 1:
        return {"b", "c"}, "a"
    return {"a", "c"}, "b"


def get_generation_subsets(
    generation: int,
    trajectory_dataset: SourGrapeDataset,
) -> tuple[Subset, Subset]:
    # Return the exposed and held-out subsets for this generation.
    exposure_labels, heldout_label = get_generation_labels(generation)
    exposure_indices = [
        idx for idx, subset in enumerate(trajectory_dataset.subsets)
        if subset in exposure_labels
    ]
    heldout_indices = [
        idx for idx, subset in enumerate(trajectory_dataset.subsets)
        if subset == heldout_label
    ]
    return (
        Subset(trajectory_dataset, exposure_indices),
        Subset(trajectory_dataset, heldout_indices),
    )


def build_trajectory_loaders(
    trajectory_dataset: SourGrapeDataset,
    exposure_subset: Subset,
    heldout_subset: Subset,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    # Build the training, test, gen, and full loaders.
    train_sampler = RepeatShuffleSampler(
        dataset_size=len(exposure_subset),
        repeats=hp.train_repeats_per_epoch,
        seed=seed,
    )
    train_loader = DataLoader(
        exposure_subset,
        batch_size=hp.batch_size,
        sampler=train_sampler,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=True),
    )
    test_loader = DataLoader(
        exposure_subset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=False),
    )
    gen_loader = DataLoader(
        heldout_subset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=False),
    )
    full_loader = DataLoader(
        trajectory_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=trajectory_dataset.get_collate_batch(augment_targets=False),
    )
    return train_loader, test_loader, gen_loader, full_loader


def run_trajectory_training(
    seed: int,
    generation: int,
    trajectory_dataset: SourGrapeDataset,
    embedding_weights: torch.Tensor | None,
    device: torch.device,
    out_dir: Path,
    resume_path: str = "",
) -> tuple[np.ndarray, list[tuple[int, str, float]]]:
    # Reproducibility.
    torch.manual_seed(seed)

    # Output directory for artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Select the subsets for this generation.
    exposure_subset, heldout_subset = get_generation_subsets(
        generation,
        trajectory_dataset,
    )

    # Build the training and test loaders.
    train_loader, test_loader, gen_loader, full_loader = build_trajectory_loaders(
        trajectory_dataset=trajectory_dataset,
        exposure_subset=exposure_subset,
        heldout_subset=heldout_subset,
        seed=seed,
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

    # Optionally resume from a checkpoint.
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
        "gen_loss": [],
    }
    history_rows = []
    for epoch in range(1, hp.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            stage="train",
            device=device,
        )
        test_loss = eval_one_epoch(
            model,
            test_loader,
            loss_fn,
            stage="train",
            device=device,
        )
        gen_loss = eval_one_epoch(
            model,
            gen_loader,
            loss_fn,
            stage="train",
            device=device,
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["gen_loss"].append(gen_loss)
        history_rows.append((epoch, "train", train_loss))
        history_rows.append((epoch, "test", test_loss))
        history_rows.append((epoch, "gen", gen_loss))
        print(
            f"epoch={epoch}, train_loss={train_loss:.6f}, "
            f"test_loss={test_loss:.6f}, gen_loss={gen_loss:.6f}"
        )
        # Save a checkpoint every five epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save a single loss plot for this run.
    loss_plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(loss_plot_path))

    # Evaluate the model on the full dataset and collect aligned predictions.
    final_loss, preds = eval_last_epoch(
        model,
        full_loader,
        loss_fn,
        device,
    )
    history["final_loss"] = final_loss
    history_rows.append((hp.epochs, "final", final_loss))
    print(f"final_loss={final_loss:.6f}")

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

    # Save a small set of example trajectory plots for this generation.
    save_trajectory_plots(trajectory_dataset, preds.numpy(), out_dir)

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
        subset_seed=seed,
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
                generation=gen,
                trajectory_dataset=trajectory_dataset,
                embedding_weights=embedding_weights,
                device=device,
                out_dir=gen_out_dir,
            )

    summary_dir = iteration_root / f"{condition}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if stage in {"all", "pretrain"}:
        # Save the combined pretraining history.
        save_history_csv(pretrain_history_by_gen, summary_dir / "pretrain_history.csv")

    if stage in {"all", "train"}:
        # Save the combined trajectory-training history.
        save_history_csv(train_history_by_gen, summary_dir / "history.csv")
        save_loss_drift_plot(
            train_history_by_gen,
            str(summary_dir / "loss_drift.png"),
        )

        # Save the trajectory drift plots grouped by exposed-set vs held-out scope.
        save_trajectory_drift_plots(
            summary_dir=summary_dir,
            trajectory_dataset=trajectory_dataset,
            preds_by_gen=preds_by_gen,
            get_generation_labels=get_generation_labels,
            padding_value=hp.padding_value,
        )

        # Save all generation predictions in one CSV file.
        save_predictions_csv(
            preds_by_gen=preds_by_gen,
            words=trajectory_dataset.words,
            item_types=trajectory_dataset.item_types,
            subsets=trajectory_dataset.subsets,
            target_lengths=[len(target) for target in trajectory_dataset.y_real],
            output_path=summary_dir / "predictions.csv",
            get_generation_labels=get_generation_labels,
        )

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
