"""Model evaluation — run inference on a DataLoader and collect predictions."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluation:
    """Evaluate a model on a DataLoader, collecting predictions and targets.

    Args:
        data_loader: DataLoader returning ``(feat, target, weight)`` batches.
        model: Trained model.
        device: Torch device.
        norm_dict: Normalization parameters, with key ``'y'`` → ``(mean, std)``.
        transform: Target transform to invert (``'LN'``, ``'LG'``, ``'E^-x'``, or None).
    """

    def __init__(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        norm_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        transform: str | None = None,
    ) -> None:
        self.data_loader = data_loader
        self.model = model
        self.device = device
        mean, std = norm_dict['y']
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.transform = transform
        self.pred, self.target = self._get_pred()

    def _get_pred(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return ``(predictions, targets)``."""
        self.model.eval()
        sum_pred: list[torch.Tensor] = []
        sum_target: list[torch.Tensor] = []

        with torch.no_grad():
            for feat, target, _ in self.data_loader:
                feat = feat.to(self.device)
                output = self.model(feat)
                target = target.to(self.device)

                # Denormalize
                output = (output * self.std + self.mean).cpu()
                target = (target * self.std + self.mean).cpu()

                output = self._target_inverse_transform(output)
                target = self._target_inverse_transform(target)

                sum_pred.append(output)
                sum_target.append(target)
                del output, target

        sum_pred = torch.cat(sum_pred)
        sum_target = torch.cat(sum_target)
        return sum_pred, sum_target

    def _target_inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Invert target transformation applied during preprocessing."""
        if self.transform == 'LN':
            return torch.exp(data)
        elif self.transform == 'LG':
            return 10 ** data
        elif self.transform == 'E^-x':
            return -torch.log(data)
        elif not self.transform:
            return data
        else:
            raise ValueError(f"Unknown target_transform: '{self.transform}'")
