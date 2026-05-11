from typing import Callable, Tuple

import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    stage: str,
    device: torch.device,
) -> float:
    # Run one training epoch.
    model.train()

    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device)
        if stage == "pretrain":
            targets = batch["y"].to(device)
        else:
            # Train the trajectory model on y_prev.
            targets = batch["y_prev"].to(device)
        preds = model(x, targets=targets)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    num_batches = max(len(dataloader), 1)
    return total_loss / num_batches


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    stage: str,
    device: torch.device,
) -> float:
    # Run one evaluation epoch.
    model.eval()

    total_loss = 0.0
    for batch in dataloader:
        x = batch["x"].to(device)
        if stage == "pretrain":
            targets = batch["y"].to(device)
        else:
            # Evaluate the trajectory model on y_prev during training.
            targets = batch["y_prev"].to(device)
        preds = model(x)
        loss = loss_fn(preds, targets)
        total_loss += loss.item()

    num_batches = max(len(dataloader), 1)
    return total_loss / num_batches


@torch.no_grad()
def eval_last_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    # Evaluate one loader and collect predictions.
    model.eval()

    total_loss = 0.0
    preds_all = []

    for batch in dataloader:
        x = batch["x"].to(device)
        # Evaluate the trajectory model on y_real at the end of training.
        y = batch["y_real"].to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        preds_all.append(preds.cpu())

    pred_matrix = torch.cat(preds_all, dim=0)
    num_batches = max(len(dataloader), 1)
    return total_loss / num_batches, pred_matrix
