"""Model saving and early stopping."""
import torch
import math
from typing import Callable

from configs.schema import ModelParams


class SaveModel:
    """Handles best-model checkpointing, regular checkpointing, and early stopping.

    Args:
        norm_dict: Normalization parameters saved alongside the model.
        param: ModelParams configuration.
        model_dir: Directory for best model checkpoints.
        ckpt_dir: Directory for periodic checkpoints.
        trace_func: Logging function for early stopping messages.
    """

    def __init__(
        self,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        param: ModelParams,
        model_dir: str,
        ckpt_dir: str,
        trace_func: Callable,
    ) -> None:
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.val_loss = math.inf
        self.norm_dict = norm_dict
        self.param = param
        self.model_dir = model_dir
        self.ckpt_dir = ckpt_dir
        self.early_stopping = EarlyStopping(
            patience=param.early_stopping.patience,
            delta=param.early_stopping.delta,
            trace_func=trace_func,
        )

    def best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save best model checkpoint if current val_loss is the best so far.

        Args:
            model: Current model.
            optimizer: Current optimizer.
            epoch: Current epoch number.
            val_loss: Current validation loss.
        """
        self.val_loss = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            state_dict = {
                'norm': self.norm_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }
            torch.save(
                state_dict,
                f"{self.model_dir}/best_model_{self.param.optim_criteria}_{self.param.time}.pth",
            )

    def regular_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Save periodic checkpoint every model_save_step epochs.

        Args:
            model: Current model.
            optimizer: Current optimizer.
            epoch: Current epoch number.
        """
        if epoch % self.param.model_save_step == 0:
            state_dict = {
                'norm': self.norm_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': self.val_loss,
            }
            digits = len(str(self.param.epoch_num))
            torch.save(
                state_dict,
                f"{self.ckpt_dir}/ckpt_{self.param.time}_{epoch:0{digits}d}.pth",
            )

    def check_early_stopping(self) -> bool:
        """Return True if training should stop early."""
        self.early_stopping(self.val_loss)
        return self.early_stopping.early_stop


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience: Number of epochs to wait before stopping.
        delta: Minimum change to qualify as improvement.
        trace_func: Logging function (default print).
    """

    def __init__(self, patience: int = 7, delta: float = 0, trace_func: Callable = print) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
