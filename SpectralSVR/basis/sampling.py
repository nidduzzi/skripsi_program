"""Sampling schemes: how a basis lays out sample points over a domain.

Periodicity and node placement are basis-specific (Fourier wants equispaced,
periodic/half-open nodes and can use the FFT; Chebyshev wants Gauss-Lobatto
nodes; wavelets have their own dyadic grids). Rather than a Fourier-centric
``periodic`` boolean threaded through the transform, a basis is given a
``SamplingScheme`` object (dependency injection) that owns node generation.

Each basis narrows the acceptable scheme type to the protocol it can actually
use (``FourierAcceptableScheme``, ``ChebyshevAcceptableScheme``), so passing an
incompatible scheme is a type error.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SamplingScheme(Protocol):
    """Base protocol: produce ``n`` sample coordinates over ``[start, stop]``."""

    @property
    def is_periodic(self) -> bool: ...

    def nodes(self, n: int, start: float, stop: float) -> torch.Tensor: ...


@runtime_checkable
class FourierAcceptableScheme(SamplingScheme, Protocol):
    """Schemes usable by a Fourier basis: equispaced, and FFT-eligible or not."""

    @property
    def supports_fft(self) -> bool: ...


@runtime_checkable
class ChebyshevAcceptableScheme(SamplingScheme, Protocol):
    """Schemes usable by a Chebyshev basis (e.g. Gauss-Lobatto nodes)."""

    @property
    def is_gauss_lobatto(self) -> bool: ...


class PeriodicUniform:
    """Equispaced nodes on the half-open ``[start, stop)`` (one full period).

    The canonical Fourier grid; FFT-eligible.
    """

    is_periodic = True
    supports_fft = True

    def nodes(self, n: int, start: float, stop: float) -> torch.Tensor:
        step = (stop - start) / n
        return torch.arange(start, start + step * n, step)[:n]


class ClosedUniform:
    """Equispaced nodes on the closed ``[start, stop]`` (endpoint included).

    Used when a faithful picture of one period is wanted (e.g. plotting); not
    FFT-eligible because the endpoint duplicates the period start.
    """

    is_periodic = False
    supports_fft = False

    def nodes(self, n: int, start: float, stop: float) -> torch.Tensor:
        return torch.linspace(start, stop, n)
