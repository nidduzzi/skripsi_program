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
