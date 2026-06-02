"""Metrics for evaluating model predictions."""
import torch


class Metrics:
    """Compute regression metrics from predictions and targets.

    Args:
        pred: Predicted values.
        target: Ground truth values.
    """

    def __init__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.pred = pred
        self.target = target

    def MAE(self, dim: int | None) -> torch.Tensor:
        """Mean Absolute Error."""
        return (self.pred - self.target).abs().mean(dim=dim)

    def MSE(self, dim: int | None) -> torch.Tensor:
        """Mean Squared Error."""
        return ((self.pred - self.target) ** 2).mean(dim=dim)

    def RMSD(self, dim: int | None) -> torch.Tensor:
        """Root Mean Squared Deviation."""
        return self.MSE(dim=dim) ** 0.5

    def S2(self, dim: int | None) -> torch.Tensor:
        """Variance of the target."""
        return ((self.target - self.target.mean(dim=0)) ** 2).mean(dim=dim)

    def R2(self, dim: int | None) -> torch.Tensor:
        """Coefficient of Determination."""
        return 1 - self.MSE(dim=dim) / self.S2(dim=dim)

    def AARD(self, dim: int | None) -> torch.Tensor:
        """Average Absolute Relative Deviation."""
        return ((self.target - self.pred) / self.target).abs().mean(dim=dim)

    def RD_each(self) -> torch.Tensor:
        """Relative deviation for each sample."""
        return (self.pred - self.target) / self.target

    def MRD(self) -> float:
        """Maximum (absolute) Relative Deviation."""
        rd_each = self.RD_each()
        max_val = torch.max(rd_each).item()
        min_val = torch.min(rd_each).item()
        if abs(max_val) > abs(min_val):
            return max_val
        else:
            return min_val
