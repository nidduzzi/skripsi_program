"""Operator baselines: a common function->function interface.

These wrap neural-operator methods (DeepONet, and later FNO/SNO/NSM) behind one
interface so they can be compared against SpectralSVR on the same problems. An
``Operator`` learns a mapping from an input function (sampled at fixed sensor
locations) to an output function it can then evaluate at arbitrary query points.
"""

from __future__ import annotations

import abc

import torch

from .dataset import OperatorDataset


class Operator(abc.ABC):
    """A learned function-to-function map, fit on an :class:`OperatorDataset`.

    Subclasses adapt a concrete method (SpectralSVR, DeepONet, ...) to this
    interface. ``name`` labels the operator in comparison tables.
    """

    name: str

    @abc.abstractmethod
    def fit(self, dataset: OperatorDataset) -> "Operator":
        """Train on ``dataset``; returns ``self`` for chaining."""
        ...

    @abc.abstractmethod
    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned operator.

        Parameters
        ----------
        f : torch.Tensor
            Shape ``(N, m)`` input functions sampled at the ``m`` training
            sensor locations.
        x : torch.Tensor
            Shape ``(Q, d)`` query points to evaluate each output function at.

        Returns
        -------
        torch.Tensor
            Shape ``(N, Q)`` predicted output values ``u(x)``.
        """
        ...

    # Comparison metrics are the project's existing suite -- ``utils.get_metrics``
    # (mse / rmse / mae / r2 / smape / rse / rrse). The benchmark scores every
    # operator's predictions with it, so all methods share the same yardstick.
