"""DeepONet baseline via deepxde (torch backend).

Wraps deepxde's ``DeepONetCartesianProd`` behind the Operator interface. deepxde
is an optional dependency (the ``baselines`` extra); it is imported lazily so the
package stays importable without it.
"""

from __future__ import annotations

import os

import torch

from .__base import Operator
from .dataset import OperatorDataset


def _import_deepxde():
    # deepxde reads its backend at import time; default it to torch. Imported
    # lazily so `operators` works without the optional `baselines` extra.
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    try:
        import deepxde as dde
        from deepxde.nn.pytorch import DeepONetCartesianProd
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "DeepONetOperator needs the 'baselines' extra: uv sync --extra baselines"
        ) from exc
    return dde, DeepONetCartesianProd


class DeepONetOperator(Operator):
    """DeepONet (unstacked, Cartesian-product) operator baseline.

    The training targets are the output functions sampled on ``trunk`` points;
    at predict time DeepONet can be queried at arbitrary points (its trunk net
    generalises across the domain).
    """

    name = "DeepONet"

    def __init__(
        self,
        trunk: torch.Tensor,
        branch_layers: tuple[int, ...] = (40, 40),
        trunk_layers: tuple[int, ...] = (40, 40),
        activation: str = "relu",
        lr: float = 1e-3,
        iterations: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.trunk = trunk.to(torch.float64)
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.lr = lr
        self.iterations = iterations
        self.seed = seed
        self._model = None

    def fit(self, dataset: OperatorDataset) -> "DeepONetOperator":
        dde, DeepONetCartesianProd = _import_deepxde()
        import numpy as np

        m = dataset.sensors.shape[0]  # branch input dimension (# sensors)
        d = int(self.trunk.shape[1])  # trunk input dimension (space/time dims)

        branch = dataset.f.detach().cpu().numpy().astype(np.float32)  # (N, m)
        trunk = self.trunk.detach().cpu().numpy().astype(np.float32)  # (Q, d)
        y = dataset.targets(self.trunk).detach().cpu().numpy().astype(np.float32)

        # deepxde requires a test split; reuse the training data (this baseline is
        # evaluated by the benchmark on its own held-out set, not by deepxde).
        data = dde.data.TripleCartesianProd(
            X_train=(branch, trunk), y_train=y, X_test=(branch, trunk), y_test=y
        )
        net = DeepONetCartesianProd(
            [m, *self.branch_layers],
            [d, *self.trunk_layers],
            self.activation,
            "Glorot normal",
        )
        dde.config.set_random_seed(self.seed)
        model = dde.Model(data, net)
        model.compile("adam", lr=self.lr)
        model.train(iterations=self.iterations, display_every=self.iterations)
        self._model = model
        return self

    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "call fit before predict"
        import numpy as np

        branch = f.detach().cpu().numpy().astype(np.float32)
        trunk = x.to(torch.float64).detach().cpu().numpy().astype(np.float32)
        pred = self._model.predict((branch, trunk))
        return torch.as_tensor(pred, dtype=torch.float64)
