"""Exact / manufactured solution families for the problem tests.

All builders are deterministic given a torch generator so they compose with
Hypothesis-drawn parameters. Grids follow the FourierBasis convention: the
spatial axis is ``linspace(0, L, ns + 1)[:-1]`` (period ``L``, ``ns`` points).
"""

import math

import torch
from hypothesis import HealthCheck, settings

SETTINGS = settings(
    deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
)


def space_grid(ns: int, length: float) -> torch.Tensor:
    return torch.linspace(0, length, ns + 1, dtype=torch.float64)[:-1]


# --------------------------------------------------------------------------- #
# Heat-mode family: u = sum_i A_i sin(k_i x + phi_i) exp(-nu k_i^2 t).
# Each term exactly solves the heat equation u_t = nu u_xx, so for Burgers the
# forcing that makes u exact is simply f = u u_x.
# --------------------------------------------------------------------------- #
def heat_mode_params(n, n_modes, seed, max_m):
    g = torch.Generator().manual_seed(seed)
    m = torch.randint(1, max_m + 1, (n, n_modes), generator=g).double()
    amp = (0.5 + torch.rand((n, n_modes), generator=g).double())
    phi = 2 * math.pi * torch.rand((n, n_modes), generator=g).double()
    return m, amp, phi


def heat_mode_u_and_ux(x, t, m, amp, phi, nu, length):
    """Return (u, u_x) of the heat-mode field at time ``t`` on grid ``x``.

    x: (ns,), returns (n, ns).
    """
    k = (2 * math.pi * m / length).unsqueeze(-1)  # (n, modes, 1)
    a = amp.unsqueeze(-1)
    ph = phi.unsqueeze(-1)
    xg = x.view(1, 1, -1)
    decay = torch.exp(-nu * k**2 * t)
    u = (a * torch.sin(k * xg + ph) * decay).sum(dim=1)
    u_x = (a * k * torch.cos(k * xg + ph) * decay).sum(dim=1)
    return u, u_x


# --------------------------------------------------------------------------- #
# Cole-Hopf family: theta solves the heat equation and u = -2 nu theta_x/theta
# is an EXACT solution of the *unforced* viscous Burgers equation.
# --------------------------------------------------------------------------- #
def cole_hopf_params(n, n_modes, seed, max_m, amp=0.2):
    g = torch.Generator().manual_seed(seed)
    m = torch.randint(1, max_m + 1, (n, n_modes), generator=g).double()
    a = amp * torch.rand((n, n_modes), generator=g).double()  # keeps theta > 0
    b = 2 * math.pi * torch.rand((n, n_modes), generator=g).double()
    return m, a, b


def cole_hopf_u(x, t, m, a, b, nu, length):
    """Exact unforced-Burgers solution at time ``t`` on grid ``x`` -> (n, ns)."""
    k = (2 * math.pi * m / length).unsqueeze(-1)  # (n, modes, 1)
    aa = a.unsqueeze(-1)
    bb = b.unsqueeze(-1)
    xg = x.view(1, 1, -1)
    decay = torch.exp(-nu * k**2 * t)
    theta = 1.0 + (aa * decay * torch.cos(k * xg + bb)).sum(dim=1)
    theta_x = (-aa * k * decay * torch.sin(k * xg + bb)).sum(dim=1)
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
    nt, ns, nu, mx=2, mt=2, amp_x=1.0, amp_t=0.7, length=1.0, t_length=1.0
):
    """Return ``(u, f)`` values of shape ``(1, nt, ns)`` for ``u(t, x)``.

    ``u = amp_x sin(kx x) + amp_t sin(wt t)``; ``f = u_t + u u_x - nu u_xx`` with
    every derivative taken analytically. Band-limited: the nonlinear ``u u_x``
    reaches wavenumber ``2*mx``, so pick ``ns > 4*mx`` and ``nt > 2*mt``.
    """
    x = space_grid(ns, length)
    t = space_grid(nt, t_length)
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
