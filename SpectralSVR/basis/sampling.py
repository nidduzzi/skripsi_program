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

# These are typing-only Protocols (no isinstance checks), so they are not
# @runtime_checkable -- that avoids the isinstance overhead. If a runtime check
# is ever needed, prefer a hasattr-based TypeGuard over runtime_checkable
# isinstance (benchmark first).


@runtime_checkable
class SamplingScheme(Protocol):
    """Base protocol: produce ``n`` sample coordinates over ``[start, stop]``.

    ``is_periodic`` reports whether the grid is half-open (one period, endpoint
    excluded); ``supports_fft`` whether a fast transform applies to this grid.
    """

    @property
    def is_periodic(self) -> bool: ...

    @property
    def supports_fft(self) -> bool: ...

    def nodes(self, n: int, start: float, stop: float) -> torch.Tensor: ...


class FourierAcceptableScheme(SamplingScheme, Protocol):
    """Schemes usable by a Fourier basis (equispaced, FFT-eligible or not)."""


class ChebyshevAcceptableScheme(SamplingScheme, Protocol):
    """Schemes usable by a Chebyshev basis (e.g. Gauss-Lobatto nodes)."""

    @property
    def is_gauss_lobatto(self) -> bool: ...


class PeriodicUniform:
    """Equispaced nodes on the half-open ``[start, stop)`` (one full period).

    The canonical Fourier grid. FFT-eligible by default; pass
    ``supports_fft=False`` to force the explicit (matmul) transform path.
    """

    is_periodic = True

    def __init__(self, supports_fft: bool = True) -> None:
        self.supports_fft = supports_fft

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


# A sampling input may be a single scheme (broadcast to every axis) or one scheme
# per axis. This mirrors the per-axis ``domain`` handling: node placement and FFT
# eligibility are per-axis (raw_transform evaluates each axis independently), so
# the sampling scheme is per-axis too. (Boundary conditions -- periodic vs
# Dirichlet/Neumann -- are a separate, future concept, independent of node
# placement; Fourier only supports the periodic boundary.)
type SamplingInputType = (
    SamplingScheme | tuple[SamplingScheme, ...] | list[SamplingScheme] | None
)


def samplings_to_tuple(
    sampling: SamplingInputType,
    ndim: int,
    default: SamplingScheme,
) -> tuple[SamplingScheme, ...]:
    """Normalise a sampling input into one scheme per axis.

    ``None`` uses ``default`` on every axis; a single scheme is broadcast; a
    per-axis sequence is taken axis-for-axis (a scheme is not a ``Sequence``, so
    the two are told apart by type).
    """
    if sampling is None:
        return tuple(default for _ in range(ndim))
    # peel the single scheme off first (runtime-checkable): a per-axis list/tuple
    # has no scheme attributes, so it won't match, leaving the sequence case.
    if isinstance(sampling, SamplingScheme):
        return tuple(sampling for _ in range(ndim))
    return tuple(scheme for scheme in sampling)
