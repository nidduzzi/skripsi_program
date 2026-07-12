"""OperatorDataset: the common data every operator baseline trains on.

Built from a problem's generated function pair, it holds the input functions
sampled at fixed sensor locations plus the output functions as an evaluable
basis field. Every adapter reads what it needs from the same object -- so
SpectralSVR (which wants the output coefficients) and DeepONet (which wants the
output sampled at trunk points) compare on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..basis import Basis, DomainAxis


def sensor_grid(domain: tuple[DomainAxis, ...], n_per_axis: int) -> torch.Tensor:
    """Half-open (periodic) tensor-product grid over ``domain``.

    Parameters
    ----------
    domain : tuple[tuple[float, float], ...]
        Per-axis ``(start, stop)`` intervals.
    n_per_axis : int
        Points per axis; the returned grid has ``n_per_axis ** len(domain)``
        rows.

    Returns
    -------
    torch.Tensor
        Shape ``(m, d)`` sensor coordinates, ``m = n_per_axis ** d``.
    """
    axes = [
        torch.linspace(start, stop, n_per_axis + 1, dtype=torch.float64)[:-1]
        for start, stop in domain
    ]
    mesh = torch.meshgrid(axes, indexing="ij")
    return torch.stack([g.flatten() for g in mesh], dim=-1)


@dataclass
class OperatorDataset:
    """Input functions at sensors + output functions as an evaluable field.

    Attributes
    ----------
    sensors : torch.Tensor
        Shape ``(m, d)`` sensor coordinates the input functions are sampled at.
    f : torch.Tensor
        Shape ``(N, m)`` real input-function values at the sensors (the operator
        input / DeepONet branch input).
    u : Basis
        The ``N`` output functions as a basis field -- evaluable at arbitrary
        query points and carrying spectral coefficients.
    """

    sensors: torch.Tensor
    f: torch.Tensor
    u: Basis

    @property
    def n_functions(self) -> int:
        return self.f.shape[0]

    @classmethod
    def from_fields(
        cls,
        input_field: Basis,
        output_field: Basis,
        n_sensors: int,
    ) -> "OperatorDataset":
        """Sample ``input_field`` at a sensor grid; keep ``output_field`` as-is.

        Parameters
        ----------
        input_field, output_field : Basis
            The operator's input and output functions (which of a problem's
            generated pair is which is the caller's choice).
        n_sensors : int
            Points per axis for the sensor grid.
        """
        sensors = sensor_grid(input_field.domain, n_sensors)
        f = input_field(sensors).real.to(torch.float64)
        return cls(sensors=sensors, f=f, u=output_field)

    def targets(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth output values ``u(x)`` at query points ``x`` -> ``(N, Q)``."""
        return self.u(x).real.to(torch.float64)
