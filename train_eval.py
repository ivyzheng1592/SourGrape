from typing import Tuple

import torch
from torch import nn

from typing import Callable


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    aux_loss_weight: float = 0.0,
    training_type: str = "train",
) -> Tuple[float, float, float]:
    # Run one training epoch.
    model.train()

    total_loss = 0.0
    main_loss_total = 0.0
    aux_loss_total = 0.0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device)
        if training_type == "pretrain":
            targets = batch["y"].to(device)
        else:
            # Train the trajectory model on y_prev.
            targets = batch["y_prev"].to(device)
        preds = model(x)
        main_loss = loss_fn(preds, targets)
        aux_loss = torch.tensor(0.0, device=device)
        if aux_loss_fn is not None and training_type != "pretrain":
            # Add the penalty loss to the trajectory loss.
            penalty_targets = batch["penalty_target"].to(device)
            aux_loss = aux_loss_fn(preds, penalty_targets)
        weighted_aux_loss = aux_loss_weight * aux_loss
        loss = main_loss + weighted_aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        main_loss_total += main_loss.item()
        aux_loss_total += weighted_aux_loss.item()

    num_batches = max(len(dataloader), 1)
    return (
        total_loss / num_batches,
        main_loss_total / num_batches,
        aux_loss_total / num_batches,
    )


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    aux_loss_weight: float = 0.0,
    training_type: str = "train",
) -> Tuple[float, float, float]:
    # Run one evaluation epoch.
    model.eval()

    total_loss = 0.0
    main_loss_total = 0.0
    aux_loss_total = 0.0

    for batch in dataloader:
        x = batch["x"].to(device)
        if training_type == "pretrain":
            targets = batch["y"].to(device)
        else:
            # Evaluate the trajectory model on y_real.
            targets = batch["y_real"].to(device)
        preds = model(x)
        main_loss = loss_fn(preds, targets)
        aux_loss = torch.tensor(0.0, device=device)
        if aux_loss_fn is not None and training_type != "pretrain":
            # Add the penalty loss to the trajectory loss.
            penalty_targets = batch["penalty_target"].to(device)
            aux_loss = aux_loss_fn(preds, penalty_targets)
        weighted_aux_loss = aux_loss_weight * aux_loss
        loss = main_loss + weighted_aux_loss
        total_loss += loss.item()
        main_loss_total += main_loss.item()
        aux_loss_total += weighted_aux_loss.item()

    num_batches = max(len(dataloader), 1)
    return (
        total_loss / num_batches,
        main_loss_total / num_batches,
        aux_loss_total / num_batches,
    )


@torch.no_grad()
def eval_last_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    aux_loss_weight: float = 0.0,
    training_type: str = "train",
) -> Tuple[float, float, float, torch.Tensor]:
    # Evaluate on the full dataset and collect predictions.
    model.eval()

    total_loss = 0.0
    main_loss_total = 0.0
    aux_loss_total = 0.0
    preds_all = []

    for batch in dataloader:
        x = batch["x"].to(device)
        if training_type == "pretrain":
            y = batch["y"].to(device)
        else:
            # Evaluate the trajectory model on y_real.
            y = batch["y_real"].to(device)
        preds = model(x)
        main_loss = loss_fn(preds, y)
        aux_loss = torch.tensor(0.0, device=device)
        if aux_loss_fn is not None and training_type != "pretrain":
            # Add the penalty loss to the trajectory loss.
            penalty_targets = batch["penalty_target"].to(device)
            aux_loss = aux_loss_fn(preds, penalty_targets)
        weighted_aux_loss = aux_loss_weight * aux_loss
        loss = main_loss + weighted_aux_loss
        total_loss += loss.item()
        main_loss_total += main_loss.item()
        aux_loss_total += weighted_aux_loss.item()
        preds_all.append(preds.cpu())

    pred_matrix = torch.cat(preds_all, dim=0)
    num_batches = max(len(dataloader), 1)
    return (
        total_loss / num_batches,
        main_loss_total / num_batches,
        aux_loss_total / num_batches,
        pred_matrix,
    )
