"""Fuzzed + MMS tests for the problem generators, including exact known
solutions for every path.

* Antiderivative: exact spectral derivative of known band-limited fields.
* Burgers MMS path: manufactured (u, f) with f = L(u); spectral residual == 0.
* Burgers numerical path: Cole-Hopf exact unforced solution reproduced by the
  ETDRK4 solver.
"""

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.problems import Burgers

from _exact import (
    SETTINGS,
    cole_hopf_params,
    cole_hopf_u,
    space_grid,
)


# =========================================================================== #
# Antiderivative
# =========================================================================== #
@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(1, 4),
    m=st.integers(1, 6),
    modes=st.integers(16, 32),
    seed=st.integers(0, 10_000),
)
def test_antiderivative_exact_derivative_of_known_field(n, m, modes, seed):
    # Exact known field u = A sin(2*pi*m*x/L); its derivative is analytic.
    length = 1.0
    g = torch.Generator().manual_seed(seed)
    amp = (0.5 + torch.rand(n, generator=g)).view(n, 1)
    x = space_grid(modes, length)
    kx = 2 * math.pi * m / length
    u_val = amp * torch.sin(kx * x).view(1, -1)
    u = FourierBasis(FourierBasis.transform(u_val + 0j), periods=length)

    du_val = u.grad().inv_transform(u.grad().coeff).real
    analytic = amp * kx * torch.cos(kx * x).view(1, -1)
    assert torch.allclose(du_val, analytic, atol=1e-3)


@pytest.mark.mms
@SETTINGS
@given(n=st.integers(1, 5), modes=st.integers(3, 16), seed=st.integers(0, 10_000))
def test_antiderivative_spectral_residual_is_zero(n, modes, seed):
    problem = Antiderivative()
    u, ut = problem.generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=0
    )
    residual = problem.spectral_residual(u, ut)
    assert residual.coeff.abs().max() < 1e-3


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_antiderivative_rejects_bad_sizes():
    problem = Antiderivative()
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 0, 8)
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 4, 0)


# =========================================================================== #
# Burgers -- MMS path (forcing chosen so u solves the equation exactly)
# =========================================================================== #
@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 16),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_mms_solution_satisfies_equation(n, space_modes, nu, seed):
    problem = Burgers()
    modes = (space_modes, space_modes)
    u, f = problem.mms_solution(
        FourierBasis, n, modes, nu=nu, periods=(1.0, 1.0),
        generator=torch.Generator().manual_seed(seed),
    )
    residual = problem.spectral_residual(u, f, nu)
    assert residual.coeff.abs().max() < 1e-4
    assert f.coeff.abs().max() > 0  # non-trivial forcing


# =========================================================================== #
# Burgers -- numerical path (ETDRK4) reproduces an exact Cole-Hopf solution
# =========================================================================== #
@pytest.mark.mms
@SETTINGS
@given(
    n_modes=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_burgers_numerical_matches_cole_hopf_exact(n_modes, seed):
    # Cole-Hopf gives an exact solution of the *unforced* viscous Burgers eqn.
    # Seed the solver with u(.,0) and compare the final state to the exact one.
    nu = 0.08
    length = 2 * math.pi
    ns = 128
    T = 1.0
    nt = 300
    n = 2
    m, a, b = cole_hopf_params(n, n_modes, seed, max_m=3)
    x = space_grid(ns, length)
    u0 = cole_hopf_u(x, 0.0, m, a, b, nu, length)  # (n, ns)

    problem = Burgers()
    u_gen, _ = problem.generate(
        FourierBasis, n, (ns, ns), nu=nu, u0=u0, f=0,
        space_domain=slice(0, length, ns), time_domain=slice(0, T, nt),
    )
    final = u_gen.inv_transform(u_gen.coeff[:, -1]).real.cpu()
    exact = cole_hopf_u(x, T, m, a, b, nu, length)
    rel = (final - exact).abs().max() / exact.abs().max()
    assert rel < 5e-2


@pytest.mark.no_mms
@pytest.mark.no_fuzz
def test_burgers_numerical_solution_dissipates_energy():
    # Viscous Burgers with zero forcing must not gain energy over time.
    problem = Burgers()
    u, _ = problem.generate(
        FourierBasis, 2, (64, 64), generator=torch.Generator().manual_seed(0),
        nu=0.05, space_domain=slice(0, 1, 64), time_domain=slice(0, 1, 100),
    )
    vals = u.get_values(res=64).real
    energy = (vals**2).mean(dim=(0, 2))
    assert energy[-1] <= energy[0] + 1e-6
