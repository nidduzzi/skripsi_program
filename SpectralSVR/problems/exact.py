"""Exact / manufactured reference solutions for the problems.

These closed-form solution families are the independent ground truth used to
verify the problem operators (``spectral_residual`` / ``residual``): the forcing
that makes each ``u`` exact is known analytically, so a residual computed from
them is a genuine check, not a round trip through the operator under test. They
are library citizens (not test-only) so they can also seed benchmarks and demos.

All builders are deterministic given a seed so they compose with fuzzed
parameters. Grids follow the FourierBasis convention: the periodic spatial axis
is ``linspace(0, L, ns + 1)[:-1]`` (period ``L``, ``ns`` points, endpoint
excluded).
"""

import math

import torch


def space_grid(ns: int, length: float) -> torch.Tensor:
    """Half-open periodic grid over one period.

    Parameters
    ----------
    ns : int
        Number of grid points (endpoint excluded).
    length : float
        Period ``L``; the grid spans ``[0, L)``.

    Returns
    -------
    torch.Tensor
        Shape ``(ns,)`` float64 node coordinates, spacing ``L / ns``.
    """
    return torch.linspace(0, length, ns + 1, dtype=torch.float64)[:-1]


# --------------------------------------------------------------------------- #
# Heat-mode family: u = sum_i A_i sin(k_i x + phi_i) exp(-nu k_i^2 t).
# Each term exactly solves the heat equation u_t = nu u_xx, so for Burgers the
# forcing that makes u exact is simply f = u u_x.
# --------------------------------------------------------------------------- #
def heat_mode_params(
    n: int, n_modes: int, seed: int, max_m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw random parameters for a sum-of-heat-modes field.

    Parameters
    ----------
    n : int
        Number of independent sample fields (batch size).
    n_modes : int
        Number of sine modes summed per field.
    seed : int
        Seed for the (reproducible) torch generator.
    max_m : int
        Maximum integer wavenumber index; each mode draws ``m in [1, max_m]``.

    Returns
    -------
    wavenumber_index : torch.Tensor
        Shape ``(n, n_modes)``. Integer mode index ``m`` (physical wavenumber
        ``k = 2*pi*m / length``).
    amplitude : torch.Tensor
        Shape ``(n, n_modes)``. Per-mode amplitude in ``[0.5, 1.5)``.
    phase : torch.Tensor
        Shape ``(n, n_modes)``. Per-mode phase in ``[0, 2*pi)``.
    """
    g = torch.Generator().manual_seed(seed)
    wavenumber_index = torch.randint(1, max_m + 1, (n, n_modes), generator=g).double()
    amplitude = 0.5 + torch.rand((n, n_modes), generator=g).double()
    phase = 2 * math.pi * torch.rand((n, n_modes), generator=g).double()
    return wavenumber_index, amplitude, phase


def heat_mode_u_and_ux(
    x: torch.Tensor,
    t: torch.Tensor,
    wavenumber_index: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    nu: float,
    length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heat-mode field and its spatial derivative on a space-time grid.

    ``u = sum_i A_i sin(k_i x + phi_i) exp(-nu k_i^2 t)`` with
    ``k_i = 2*pi*m_i / length`` exactly solves the heat equation
    ``u_t = nu u_xx``; the Burgers forcing that makes it exact is ``f = u u_x``.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(ns,)`` spatial grid (e.g. from :func:`space_grid`).
    t : torch.Tensor
        Shape ``(nt,)`` time grid; the field is evaluated at every ``(t, x)``.
    wavenumber_index, amplitude, phase : torch.Tensor
        Shape ``(n, n_modes)`` per-mode parameters from
        :func:`heat_mode_params`.
    nu : float
        Viscosity (heat-diffusion coefficient).
    length : float
        Spatial period ``L``.

    Returns
    -------
    u : torch.Tensor
        Shape ``(n, nt, ns)`` field values.
    u_x : torch.Tensor
        Shape ``(n, nt, ns)`` analytic spatial derivative.
    """
    k = (2 * math.pi * wavenumber_index / length).view(*wavenumber_index.shape, 1, 1)
    a = amplitude.view(*amplitude.shape, 1, 1)
    ph = phase.view(*phase.shape, 1, 1)
    xg = x.view(1, 1, 1, -1)  # (1, 1, 1, ns)
    tg = t.view(1, 1, -1, 1)  # (1, 1, nt, 1)
    decay = torch.exp(-nu * k**2 * tg)  # (n, modes, nt, 1)
    u = (a * torch.sin(k * xg + ph) * decay).sum(dim=1)  # (n, nt, ns)
    u_x = (a * k * torch.cos(k * xg + ph) * decay).sum(dim=1)
    return u, u_x


# --------------------------------------------------------------------------- #
# Cole-Hopf family: theta solves the heat equation and u = -2 nu theta_x/theta
# is an EXACT solution of the *unforced* viscous Burgers equation.
# --------------------------------------------------------------------------- #
def cole_hopf_params(
    n: int, n_modes: int, seed: int, max_m: int, amp: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw random parameters for a Cole-Hopf potential ``theta``.

    ``theta = 1 + sum_i a_i cos(k_i x + b_i) exp(-nu k_i^2 t)`` solves the heat
    equation; the ``amp`` cap keeps ``theta > 0`` so ``u = -2 nu theta_x/theta``
    stays finite.

    Parameters
    ----------
    n : int
        Number of independent sample fields (batch size).
    n_modes : int
        Number of cosine modes in the potential ``theta``.
    seed : int
        Seed for the (reproducible) torch generator.
    max_m : int
        Maximum integer wavenumber index; each mode draws ``m in [1, max_m]``.
    amp : float, optional
        Upper bound on per-mode amplitude (default 0.2), kept small so
        ``theta`` stays strictly positive.

    Returns
    -------
    wavenumber_index : torch.Tensor
        Shape ``(n, n_modes)``. Integer mode index ``m``.
    amplitude : torch.Tensor
        Shape ``(n, n_modes)``. Per-mode amplitude in ``[0, amp)``.
    phase : torch.Tensor
        Shape ``(n, n_modes)``. Per-mode phase in ``[0, 2*pi)``.
    """
    g = torch.Generator().manual_seed(seed)
    wavenumber_index = torch.randint(1, max_m + 1, (n, n_modes), generator=g).double()
    amplitude = amp * torch.rand((n, n_modes), generator=g).double()
    phase = 2 * math.pi * torch.rand((n, n_modes), generator=g).double()
    return wavenumber_index, amplitude, phase


def cole_hopf_u(
    x: torch.Tensor,
    t: torch.Tensor,
    wavenumber_index: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    nu: float,
    length: float,
) -> torch.Tensor:
    """Exact *unforced* viscous-Burgers solution on a space-time grid.

    Via the Cole-Hopf transform ``u = -2 nu theta_x / theta``, with ``theta`` a
    heat-equation solution, this solves ``u_t + u u_x = nu u_xx`` exactly with
    zero forcing.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(ns,)`` spatial grid.
    t : torch.Tensor
        Shape ``(nt,)`` time grid; the field is evaluated at every ``(t, x)``.
    wavenumber_index, amplitude, phase : torch.Tensor
        Shape ``(n, n_modes)`` potential parameters from
        :func:`cole_hopf_params`.
    nu : float
        Viscosity.
    length : float
        Spatial period ``L``.

    Returns
    -------
    torch.Tensor
        Shape ``(n, nt, ns)`` solution values ``u(t, x)``.
    """
    k = (2 * math.pi * wavenumber_index / length).view(*wavenumber_index.shape, 1, 1)
    a = amplitude.view(*amplitude.shape, 1, 1)
    b = phase.view(*phase.shape, 1, 1)
    xg = x.view(1, 1, 1, -1)  # (1, 1, 1, ns)
    tg = t.view(1, 1, -1, 1)  # (1, 1, nt, 1)
    decay = torch.exp(-nu * k**2 * tg)  # (n, modes, nt, 1)
    theta = 1.0 + (a * decay * torch.cos(k * xg + b)).sum(dim=1)  # (n, nt, ns)
    theta_x = (-a * k * decay * torch.sin(k * xg + b)).sum(dim=1)
    return -2 * nu * theta_x / theta


# --------------------------------------------------------------------------- #
# Space-time periodic manufactured Burgers field. Unlike Cole-Hopf / heat-mode
# (which decay in t and so are non-periodic in time), this is periodic on both
# axes, so it is represented exactly by a 2D Fourier field and its *spectral*
# time derivative is exact -- the case ``Burgers.spectral_residual`` needs. The
# forcing ``f`` is built from ANALYTIC derivatives (never from the operator under
# test), so residual(u, f) == 0 is an independent check.
# --------------------------------------------------------------------------- #
def burgers_periodic_manufactured(
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float,
    mx: int = 2,
    mt: int = 2,
    amp_x: float = 1.0,
    amp_t: float = 0.7,
    length: float = 1.0,
    t_length: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Space-time periodic manufactured Burgers solution and its exact forcing.

    ``u(t, x) = amp_x sin(kx x) + amp_t sin(wt t)`` with ``kx = 2*pi*mx/length``
    and ``wt = 2*pi*mt/t_length``. The forcing ``f = u_t + u u_x - nu u_xx`` is
    formed from analytic derivatives, so ``residual(u, f) == 0`` is an
    independent check. Periodic on both axes (unlike the decaying Cole-Hopf /
    heat-mode families), so a *spectral* time derivative is exact.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(ns,)`` spatial grid over ``[0, length)`` (e.g. from
        :func:`space_grid`).
    t : torch.Tensor
        Shape ``(nt,)`` time grid over ``[0, t_length)``.
    nu : float
        Viscosity.
    mx, mt : int, optional
        Spatial / temporal integer wavenumber indices (default 2, 2). The
        nonlinear ``u u_x`` reaches wavenumber ``2*mx``, so keep ``ns > 4*mx``
        and ``nt > 2*mt`` to stay band-limited (alias-free).
    amp_x, amp_t : float, optional
        Spatial / temporal amplitudes (default 1.0, 0.7).
    length, t_length : float, optional
        Spatial / temporal periods the wavenumbers are built from
        (default 1.0, 1.0).

    Returns
    -------
    u : torch.Tensor
        Shape ``(1, nt, ns)`` solution values on the space-time grid.
    f : torch.Tensor
        Shape ``(1, nt, ns)`` exact Burgers forcing for ``u``.
    """
    ns, nt = x.shape[0], t.shape[0]
    kx = 2 * math.pi * mx / length
    wt = 2 * math.pi * mt / t_length
    X = x.view(1, 1, ns)
    T = t.view(1, nt, 1)
    u = amp_x * torch.sin(kx * X) + amp_t * torch.sin(wt * T)  # (1, nt, ns)
    u_t = (amp_t * wt * torch.cos(wt * T)).expand(1, nt, ns)
    u_x = (amp_x * kx * torch.cos(kx * X)).expand(1, nt, ns)
    u_xx = (-amp_x * kx * kx * torch.sin(kx * X)).expand(1, nt, ns)
    f = u_t + u * u_x - nu * u_xx
    return u.contiguous(), f.contiguous()
