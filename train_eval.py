from typing import Dict

import torch
from torch import nn


def _step_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: nn.Module,
) -> torch.Tensor:
    # Move batch to device and compute loss.
    x = batch["x"].to(device)
    targets = batch["y"].to(device)
    preds = model(x)
    return loss_fn(preds, targets)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    # One full pass over the training set.
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Standard training step.
        optimizer.zero_grad(set_to_none=True)
        loss = _step_batch(model, batch, device, loss_fn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    # One full pass over the validation/test set.
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Forward-only evaluation.
        x = batch["x"].to(device)
        targets = batch["y"].to(device)

        preds = model(x)
        loss = loss_fn(preds, targets)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
