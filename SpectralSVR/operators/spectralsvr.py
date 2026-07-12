"""SpectralSVR as an :class:`Operator` (the method under comparison)."""

from __future__ import annotations

import torch

from ..basis import Basis
from ..model import SpectralSVR
from ..model.__base import MultiRegression
from .__base import Operator
from .dataset import OperatorDataset


class SpectralSVROperator(Operator):
    """Adapter exposing the SpectralSVR pipeline through the Operator interface.

    Trains the LSSVR to map sensor-sampled input functions to the output spectral
    coefficients, then evaluates the reconstructed output at query points.
    """

    name = "SpectralSVR"

    def __init__(self, basis: Basis, regressor: MultiRegression) -> None:
        self._basis = basis
        self._regressor = regressor
        self._model: SpectralSVR | None = None

    def fit(self, dataset: OperatorDataset) -> "SpectralSVROperator":
        model = SpectralSVR(self._basis, self._regressor)
        model.train(
            dataset.f, dataset.u.coeff, u_time_dependent=dataset.u.time_dependent
        )
        self._model = model
        return self

    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "call fit before predict"
        return self._model.forward(f, x).real.to(torch.float64)
