from typing import Dict, Tuple

import torch
from torch import nn


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

    for batch in dataloader:
        # Standard training step.
        optimizer.zero_grad(set_to_none=True)
        # Move batch to device and compute loss.
        x = batch["x"].to(device)
        targets = batch["y_prev"].to(device)
        preds = model(x)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


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

    for batch in dataloader:
        # Forward-only evaluation.
        x = batch["x"].to(device)
        targets = batch["y_real"].to(device)

        preds = model(x)
        loss = loss_fn(preds, targets)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def eval_last_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, torch.Tensor]:
    # Evaluate on the full test set and record predictions.
    model.eval()

    total_loss = 0.0
    preds_all = []

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y_real"].to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        preds_all.append(preds.cpu())

    pred_matrix = torch.cat(preds_all, dim=0)
    return total_loss / max(len(dataloader), 1), pred_matrix
