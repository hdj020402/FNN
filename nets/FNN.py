import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class FNN(nn.Module):
    def __init__(self, dims: Iterable[int]) -> None:
        super(FNN, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, data):
        x = data.feature
        out = self.sequential_layers(x)
        return out
