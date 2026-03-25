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
from util import (
    save_loss_plot,
    save_prediction_plot,
    save_mean_trajectory_drift,
    save_item_type_error_curves,
)


def iterate_once(
    hp: HyperParams,
    generation: int,
    seed: int,
    dataset: SourGrapeDataset,
    condition: str,
    model_type: str,
    device: torch.device,
    resume_path: str = "",
) -> None:

    # Reproducibility.
    torch.manual_seed(seed)

    # Dataset and dataloaders.
    ds_a, ds_b, ds_c = dataset.split_dataset(seed=seed)

    # Rotate train split by generation number.
    mod = generation % 3
    if mod == 0:
        train_ds = torch.utils.data.ConcatDataset([ds_a, ds_b])
    elif mod == 1:
        train_ds = torch.utils.data.ConcatDataset([ds_b, ds_c])
    else:
        train_ds = torch.utils.data.ConcatDataset([ds_c, ds_a])

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False)

    # Model selection.
    if model_type == "seq2seq":
        model = Seq2SeqRegressor(
            input_size=len(dataset.vocab),
            output_len=dataset.trajectory_len,
        )
    else:
        model = LSTMRegressor(
            input_size=len(dataset.vocab),
            output_size=dataset.trajectory_len,
        )
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = nn.MSELoss(reduction="sum")

    # Optionally resume from a checkpoint.
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Output directory for artifacts.
    out_dir = Path("output") / condition / f"gen_{generation}"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    history = {
        "train_loss": [],
        "test_loss": [],
    }
    for epoch in range(1, hp.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        test_loss = eval_one_epoch(model, test_loader, device, loss_fn)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(f"epoch={epoch}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        # Save a checkpoint every 5 epochs.
        if epoch % 5 == 0:
            ckpt_path = model_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save a single loss plot for this run.
    plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(plot_path))

    # Final evaluation and prediction capture for next generation.
    final_loss, preds = eval_last_epoch(model, test_loader, device, loss_fn)
    history["final_test_loss"] = final_loss
    print(f"final_test_loss={final_loss:.6f}")
    prev_before = dataset.y_prev.mean().item()
    dataset.update_prev_targets(preds)
    prev_after = dataset.y_prev.mean().item()
    print(f"y_prev_mean_before={prev_before:.6f}, y_prev_mean_after={prev_after:.6f}")

    # Save one prediction plot per item type.
    seen_types = set()
    for idx in range(len(dataset)):
        item_type = dataset[idx]["item_type"]
        if item_type in seen_types:
            continue
        word = "".join(dataset.id_to_char[i] for i in dataset[idx]["x"].tolist())
        target = dataset[idx]["y_real"].tolist()
        prediction = preds[idx].tolist()
        pred_path = out_dir / f"prediction_vs_target_{item_type}.png"
        save_prediction_plot(word, target, prediction, str(pred_path))
        seen_types.add(item_type)
        if len(seen_types) >= 5:
            break

    # Save artifacts.
    torch.save(model.state_dict(), model_dir / "model.pt")
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(dataset.vocab, f, ensure_ascii=True, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
    np.save(out_dir / "predictions.npy", preds.numpy())


def iterate_multi(condition: str, num_generations: int, base_seed: int = 42) -> None:
    # Train multiple generations with different random seeds.
    hp = HyperParams()
    dataset = SourGrapeDataset(condition=condition)
    model_type = hp.model_type
    device = torch.device(hp.device)
    preds_by_gen = {}
    for gen in range(0, num_generations):
        iterate_once(
            hp=hp,
            generation=gen,
            seed=base_seed + gen,
            dataset=dataset,
            condition=condition,
            model_type=model_type,
            device=device,
        )
        pred_path = Path("output") / condition / f"gen_{gen}" / "predictions.npy"
        if pred_path.exists():
            preds_by_gen[gen] = np.load(pred_path)

    # Save mean trajectory drift plots across generations.
    output_dir = str(Path("output") / condition / "drift_plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_mean_trajectory_drift(preds_by_gen, dataset.item_types, output_dir)

    # Save per-item-type MSE for each generation (preds vs real).
    for gen, preds in preds_by_gen.items():
        save_item_type_error_curves(
            preds=preds,
            targets=dataset.y_real.numpy(),
            item_types=dataset.item_types,
            output_dir=output_dir,
            suffix=f"gen_{gen}",
        )


if __name__ == "__main__":
    iterate_multi(num_generations=5, condition="glide")
    iterate_multi(num_generations=5, condition="fricative")
