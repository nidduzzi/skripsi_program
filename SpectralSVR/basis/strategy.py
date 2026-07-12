"""Evaluation/transform strategy: a basis-agnostic tuning object.

Rather than expose Fourier-specific knobs (chunk sizes, NUFFT thresholds), the
user states *intent* -- a peak memory budget and whether approximate fast paths
are allowed -- and the strategy translates that into concrete decisions
(``chunk_points`` and ``use_approximate``). Those are plain methods, so a
basis-specific subclass can override them for finer control while a basis only
ever calls the generic interface.

Held on the instance (injected, like the sampling scheme) with a module
default, and threaded into the static transform methods so per-instance
overrides reach them (a class attribute could not: transform() is static).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationStrategy:
    """How much memory/accuracy to trade in non-uniform evaluation.

    - memory_budget_mb: target peak working memory for one evaluation/transform
      block; the exact path is chunked to fit it. ``float("inf")`` = no bound.
    - allow_approximate: whether the basis may use an approximate fast path
      (e.g. a NUFFT) when the exact work would exceed the budget.
    """

    memory_budget_mb: float = 256.0
    allow_approximate: bool = True

    def max_elements(self, itemsize_bytes: int) -> float:
        """Element budget: how many entries of the given size fit the budget."""
        if math.isinf(self.memory_budget_mb):
            return math.inf
        return self.memory_budget_mb * (2**20) / max(1, itemsize_bytes)

    def chunk_points(self, per_point_elems: int, itemsize_bytes: int) -> int:
        """Points per block so a block's working set fits the budget.

        ``per_point_elems`` is how many entries each evaluation point costs
        (e.g. prod(modes) for a dense basis matrix). Override for finer control.
        """
        budget = self.max_elements(itemsize_bytes)
        if math.isinf(budget):
            return 2**62  # effectively unbounded -> a single block
        return max(1, int(budget // max(1, per_point_elems)))

    def use_approximate(self, dense_elems: float, itemsize_bytes: int) -> bool:
        """Whether to take the approximate fast path for this problem size."""
        return self.allow_approximate and dense_elems > self.max_elements(
            itemsize_bytes
        )


DEFAULT_EVALUATION_STRATEGY = EvaluationStrategy()
