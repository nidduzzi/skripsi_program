"""NUFFT-backed evaluation of a Fourier series at non-uniform points.

The dense path evaluates ``value(x_j) = sum_k coeff_k exp(2*pi*i*k*x_j/L)`` by
materializing a ``points x modes`` basis matrix -- O(points * modes) memory and
time, which blows up for wide evaluation grids (e.g. 1 x 400000). A NUFFT
computes the same sum in ~O(prod(modes) log + points) via Kaiser-Bessel
gridding (torchkbnufft), at the cost of a small, tunable approximation.

Convention mapping to torchkbnufft's ``KbNufft`` (validated against the exact
matmul path): reorder each coefficient axis from fft order to natural
(ascending wavenumber) order, feed ``ktraj = -2*pi*x/period`` per dimension, and
divide the result by ``prod(modes)`` to match ``FourierBasis.evaluate``.
"""

from __future__ import annotations

import math

import torch


def nufft_available() -> bool:
    try:
        import torchkbnufft  # noqa: F401
    except ImportError:
        return False
    return True


def nufft_evaluate(
    coeff: torch.Tensor, x: torch.Tensor, periods: tuple[float, ...]
) -> torch.Tensor:
    """Evaluate a (batched) Fourier series at non-uniform points via NUFFT.

    Arguments:
        coeff {torch.Tensor} -- (batch, *modes) complex spectrum in fft order.
        x {torch.Tensor} -- (npts, ndim) evaluation points.
        periods {tuple[float, ...]} -- spatial period per dimension.

    Returns:
        torch.Tensor -- (batch, npts) complex values, matching
        FourierBasis.evaluate.
    """
    import torchkbnufft as tkbn

    modes = tuple(coeff.shape[1:])
    ndim = len(modes)
    device = coeff.device

    # fft order -> natural (ascending wavenumber) order per axis
    image = coeff
    for dim, m in enumerate(modes):
        order = torch.argsort(torch.fft.fftfreq(m, device=device))
        image = image.index_select(dim + 1, order)
    image = image.unsqueeze(1)  # (batch, 1, *modes)

    ktraj = torch.stack(
        [(-2 * math.pi * x[:, d] / periods[d]) for d in range(ndim)]
    ).to(device=device, dtype=x.real.dtype if x.is_complex() else x.dtype)

    nufft = tkbn.KbNufft(im_size=modes).to(device=device, dtype=image.dtype)
    values = nufft(image, ktraj).squeeze(1)  # (batch, npts)
    return values / math.prod(modes)
