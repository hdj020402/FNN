"""Seed setup for reproducibility."""
import torch
import random
import numpy as np


def setup_seed(seed: int, use_deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        use_deterministic: If True, enable deterministic CUDA algorithms
            (may be slower). If False, allow non-deterministic kernels.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = use_deterministic
    torch.backends.cudnn.benchmark = not use_deterministic
