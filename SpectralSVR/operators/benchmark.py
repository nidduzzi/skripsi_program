"""Run a set of operators on the same data and tabulate the comparison."""

from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd
import torch

from ..utils import get_metrics
from .__base import Operator
from .dataset import OperatorDataset


def benchmark(
    operators: Sequence[Operator],
    train: OperatorDataset,
    test: OperatorDataset,
    query: torch.Tensor,
) -> pd.DataFrame:
    """Fit each operator on ``train`` and score it on ``test`` at ``query``.

    Every operator is scored with the project's shared metric suite
    (:func:`utils.get_metrics`) against the same targets, so the columns are
    directly comparable across methods.

    Parameters
    ----------
    operators : Sequence[Operator]
        The baselines to compare (e.g. ``SpectralSVROperator``, ``DeepONetOperator``).
    train, test : OperatorDataset
        Disjoint train / test splits (same problem, same sensors).
    query : torch.Tensor
        Shape ``(Q, d)`` points to evaluate the predicted output functions at.

    Returns
    -------
    pandas.DataFrame
        One row per operator: ``operator``, ``fit_seconds``, then every metric
        from :func:`utils.get_metrics`.
    """
    # score every operator on the same host device so the metric suite never
    # trips over a GPU-trained prediction vs a CPU target
    target = test.targets(query).detach().cpu()
    rows = []
    for op in operators:
        start = time.perf_counter()
        op.fit(train)
        fit_seconds = time.perf_counter() - start
        pred = op.predict(test.f, query).detach().cpu()
        rows.append(
            {"operator": op.name, "fit_seconds": fit_seconds, **get_metrics(pred, target)}
        )
    return pd.DataFrame(rows).set_index("operator")
