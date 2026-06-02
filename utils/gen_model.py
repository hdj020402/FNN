"""Model, optimizer, and scheduler factory — thin wrappers around models/factory.py."""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.schema import ModelParams
from models.factory import create_model


def gen_model(param: ModelParams, dataset, device: torch.device | None = None) -> torch.nn.Module:
    """Create and device-move the FNN model from config.

    Args:
        param: ModelParams configuration.
        dataset: FeatureDataset.
        device: Torch device (auto-detected if None).

    Returns:
        Model moved to the appropriate device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(param, dataset)
    return model.to(device)


def gen_optimizer(param: ModelParams, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from config.

    Args:
        param: ModelParams configuration.
        model: The model whose parameters will be optimized.

    Returns:
        Optimizer instance.
    """
    return getattr(torch.optim, param.optimizer)(model.parameters(), lr=param.lr)


def gen_scheduler(param: ModelParams, optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau | None:
    """Create learning rate scheduler from config.

    Args:
        param: ModelParams configuration.
        optimizer: The optimizer to schedule.

    Returns:
        Scheduler instance, or None if type is not ReduceLROnPlateau.
    """
    if param.scheduler.type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer, mode='min',
            factor=param.scheduler.factor,
            patience=param.scheduler.patience,
            min_lr=param.scheduler.min_lr,
        )
    return None
