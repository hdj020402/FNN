"""Training and validation loops."""
from typing import Literal
from torch.utils.data import DataLoader
import torch


def weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_type: Literal['MAE', 'MSE'],
) -> torch.Tensor:
    """Compute weighted loss.

    Args:
        pred: Model predictions [B, T].
        target: Ground truth targets [B, T].
        weight: Per-sample weights [B, 1] or [B, T].
        loss_type: 'MAE' or 'MSE'.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == 'MAE':
        return (weight * torch.abs(pred - target)).mean()
    elif loss_type == 'MSE':
        return (weight * (pred - target) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'")


def unweighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal['MAE', 'MSE'],
) -> torch.Tensor:
    """Compute unweighted loss (for validation).

    Args:
        pred: Model predictions [B, T].
        target: Ground truth targets [B, T].
        loss_type: 'MAE' or 'MSE'.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == 'MAE':
        return torch.abs(pred - target).mean()
    elif loss_type == 'MSE':
        return ((pred - target) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'")


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Literal['MAE', 'MSE'],
    device: torch.device,
    accumulation_steps: int = 1,
    grad_clip_norm: float | None = None,
) -> float:
    """Train the model for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader returning ``(feat, target, weight)`` batches.
        optimizer: Optimizer.
        loss_fn: Loss function name ('MAE' or 'MSE').
        device: Torch device.
        accumulation_steps: Number of batches between optimizer steps.
        grad_clip_norm: Max gradient norm for clipping (None = disabled).

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    loss_all = 0.0
    num_batches = 0

    for i, (feat, target, weight) in enumerate(train_loader):
        feat = feat.to(device)
        target = target.to(device)
        weight = weight.to(device)
        output = model(feat)
        loss = weighted_loss(output, target, weight, loss_fn)
        loss_all += loss.item() * target.size(0)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        num_batches = i + 1

    if num_batches > 0 and num_batches % accumulation_steps != 0:
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Literal['MAE', 'MSE'],
    device: torch.device,
) -> float:
    """Validate the model.

    Args:
        model: Model to evaluate.
        loader: Data loader returning ``(feat, target, weight)`` batches.
        loss_fn: Loss function name.
        device: Torch device.

    Returns:
        Average validation loss.
    """
    model.eval()
    loss_all = 0.0

    for feat, target, weight in loader:
        feat = feat.to(device)
        target = target.to(device)
        output = model(feat)
        loss = unweighted_loss(output, target, loss_fn)
        loss_all += loss.item() * target.size(0)

    return loss_all / len(loader.dataset)
