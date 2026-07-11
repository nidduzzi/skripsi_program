"""Shared helpers for the model tests.

Manufactured-data builders (for MMS-style tests) and a common Hypothesis
settings profile. Builders are deterministic given a seed so they compose with
Hypothesis-drawn parameters.
"""

import torch
from hypothesis import HealthCheck, settings

SETTINGS = settings(
    deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
)


def linear_dataset(
    n: int, d: int, o: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manufacture y = X W + b for a known random linear map."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    W = torch.randn(d, o, generator=g)
    b = torch.randn(o, generator=g)
    return X, X @ W + b


def constant_dataset(
    n: int, d: int, o: int, const: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manufacture a constant target y = const."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    return X, torch.full((n, o), const)


def smooth_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Manufacture a smooth scalar target y = sin(sum(x))."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g)
    return X, torch.sin(X.sum(dim=1, keepdim=True))
