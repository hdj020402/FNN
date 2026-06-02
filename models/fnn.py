"""Feedforward Neural Network (FNN) model."""
import torch
import torch.nn as nn
from typing import Iterable


class FNN(nn.Module):
    """A simple feedforward neural network with ReLU activations.

    Args:
        dims: Layer dimensions, e.g. ``[input_dim, hidden_1, ..., output_dim]``.
    """

    def __init__(self, dims: Iterable[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims_list = list(dims)
        for i in range(len(dims_list) - 1):
            layers.append(nn.Linear(dims_list[i], dims_list[i + 1]))
            if i < len(dims_list) - 2:
                layers.append(nn.ReLU())
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape ``[batch_size, num_features]``.

        Returns:
            Output tensor of shape ``[batch_size, num_targets]``.
        """
        return self.sequential_layers(x)
