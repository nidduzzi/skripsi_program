"""Plotting for basis functions.

Kept out of ``__base`` so the core :class:`Basis` contract stays free of the
matplotlib dependency and easy to read when implementing a new basis.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import torch
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from .__base import AutoEvaluationModeType, Basis, ResType
    from .FourierBasis import FourierBasis

logger = logging.getLogger(__name__)


def _resolve_canvas(plt: "ModuleType | Axes | None") -> "ModuleType | Axes":
    if plt is not None:
        return plt
    import matplotlib.pyplot as pyplot  # noqa: PLC0415

    return pyplot


def plot_basis(  # noqa: C901
    basis: "Basis",
    i: int = 0,
    n: int = 1,
    res: "ResType | None" = None,
    plt: "ModuleType | Axes | None" = None,
    complex_scatter: bool = False,
    plot_component: None | Literal["imag", "real"] = None,
    legend: bool = True,
    evaluation_mode: "AutoEvaluationModeType" = "auto",
    device: "torch.device | None" = None,
    **kwargs,
):
    """Draw the functions in ``basis``; see :meth:`Basis.plot`."""
    plt = _resolve_canvas(plt)

    assert i + n <= len(basis), (
        f"values of i={i} and n={n} is out of bounds. i+n needs to be less than "
        f"or equal to the number of functions {len(basis)}"
    )
    if res is None:
        res = tuple(slice(0, period, 200) for period in basis.periods)
    plot_dims = basis.ndim + 1 if basis.time_dependent else basis.ndim
    values, grid = basis.get_values_and_grid(
        i=i, n=n, res=res, evaluation_mode=evaluation_mode, device=device
    )
    assert len(values) > 0 or values is None, (
        "something went wrong in computing the values"
    )
    values = values.cpu()
    grid = grid.cpu()

    match plot_dims:
        case 1:
            if basis._complex_funcs:
                if complex_scatter:
                    for func in values:
                        func_flat = func.flatten()
                        plot = plt.scatter(func_flat.real, func_flat.imag, **kwargs)
                    if legend:
                        plt.legend([f"Function ({i + j})" for j in range(len(values))])
                else:
                    match plot_component:
                        case "real":
                            for func in values:
                                plot = plt.plot(
                                    grid.flatten(), func.flatten().real, **kwargs
                                )
                            if legend:
                                plt.legend(
                                    [
                                        f"Real function ({i + j})"
                                        for j in range(len(values))
                                    ]
                                )
                        case "imag":
                            for func in values:
                                plot = plt.plot(
                                    grid.flatten(),
                                    func.flatten().imag,
                                    linestyle="dashed",
                                    **kwargs,
                                )
                            if legend:
                                plt.legend(
                                    [
                                        f"Imaginary function ({i + j})"
                                        for j in range(len(values))
                                    ]
                                )
                        case _:
                            for func in values:
                                func_flat = func.flatten()
                                plot = plt.plot(
                                    grid.flatten(), func_flat.real, **kwargs
                                )
                                kwargs["color"] = kwargs.get(
                                    "color", plot[0].get_color()
                                )
                                kwargs["linestyle"] = kwargs.get("linestyle", "dashed")
                                plot = plt.plot(
                                    grid.flatten(), func_flat.imag, **kwargs
                                )
                            if legend:
                                plt.legend(
                                    [
                                        f"Real function ({i + j})"
                                        if k == 0
                                        else f"Imaginary function ({i + j})"
                                        for k in range(2)
                                        for j in range(len(values))
                                    ]
                                )
            else:
                for func in values:
                    plot = plt.plot(grid.flatten(), func.flatten().real, **kwargs)
                if legend:
                    plt.legend(
                        [(f"Real function ({i + j})") for j in range(len(values))]
                    )
        case 2:
            if complex_scatter:
                for func in values:
                    func_flat = func.flatten()
                    plot = plt.scatter(func_flat.real, func_flat.imag, **kwargs)
                if legend:
                    plt.legend([f"Function ({i + j})" for j in range(len(values))])
            else:
                if plot_component is None:
                    plot_component = "real"
                    if basis._complex_funcs:
                        logger.warning("plotting only real component")
                # imshow places pixel *edges* at the extent bounds while the grid
                # holds pixel *centres*, so expand the extent by half a pixel on
                # each side to align the image with the sample coordinates.
                x0, x1 = grid[0, 0, 1].item(), grid[0, -1, 1].item()
                y0, y1 = grid[0, 0, 0].item(), grid[-1, 0, 0].item()
                half_dx = (x1 - x0) / (2 * (grid.shape[1] - 1)) if grid.shape[1] > 1 else 0.0
                half_dy = (y1 - y0) / (2 * (grid.shape[0] - 1)) if grid.shape[0] > 1 else 0.0
                extent = (x0 - half_dx, x1 + half_dx, y0 - half_dy, y1 + half_dy)
                xlim = (x0, x1)
                ylim = (y0, y1)
                kwargs["extent"] = kwargs.get("extent", extent)
                kwargs["origin"] = kwargs.get("origin", "lower")
                kwargs["aspect"] = kwargs.get("aspect", "auto")
                match plot_component:
                    case "imag":
                        plot = plt.imshow(values[0].imag, **kwargs)
                    case "real":
                        plot = plt.imshow(values[0].real, **kwargs)
                    case _:  # pragma: no cover  (plot_component is real/imag here)
                        raise NotImplementedError(
                            "Can't plot both imaginary and real in 2D"
                        )

                if isinstance(plt, Axes):
                    plt.set_xlim(*xlim)
                    plt.set_ylim(*ylim)
                else:
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)

        case _:  # pragma: no cover  (at most 2D is supported)
            raise NotImplementedError("plots for dimensions > 2 need to be implemented")

    return plot


def _component(coeff: torch.Tensor, component: str) -> torch.Tensor:
    match component:
        case "magnitude":
            return coeff.abs()
        case "real":
            return coeff.real
        case "imag":
            return coeff.imag
        case _:  # pragma: no cover  (component is a validated Literal)
            raise ValueError(f"unknown component {component!r}")


def plot_fourier_coefficients(
    basis: "FourierBasis",
    i: int = 0,
    n: int = 1,
    plt: "ModuleType | Axes | None" = None,
    component: Literal["magnitude", "real", "imag"] = "magnitude",
    legend: bool = True,
    **kwargs,
):
    """Plot the Fourier spectrum of ``n`` functions from ``i``; see
    :meth:`FourierBasis.plot_coefficients`."""
    plt = _resolve_canvas(plt)
    if basis.time_dependent:
        raise NotImplementedError(
            "plot_coefficients is not implemented for time-dependent bases"
        )
    coeff = basis.coeff[i : i + n].detach().cpu()
    modes = basis.modes

    if len(modes) == 1:
        wn = basis.wave_number(modes[0]).flatten().cpu()
        order = torch.argsort(wn)
        for c in coeff:
            plot = plt.plot(wn[order], _component(c[order], component), **kwargs)
        if legend:
            plt.legend([f"Function ({i + j})" for j in range(len(coeff))])
    elif len(modes) == 2:
        # fft-shift so the zero frequency sits at the centre of the image
        shifted = torch.fft.fftshift(coeff[0])
        kwargs["origin"] = kwargs.get("origin", "lower")
        kwargs["aspect"] = kwargs.get("aspect", "auto")
        plot = plt.imshow(_component(shifted, component), **kwargs)
    else:
        raise NotImplementedError(
            "plot_coefficients supports 1D and 2D Fourier bases"
        )
    return plot
