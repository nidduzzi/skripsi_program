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


# --------------------------------------------------------------------------- #
# Additional coverage
# --------------------------------------------------------------------------- #
@pytest.mark.no_mms
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 16),
    seed=st.integers(0, 10_000),
)
def test_burgers_manufactured_branch_shapes(n, space_modes, seed):
    # u0="random", f="random" -> spectral manufactured-solution branch
    u, f = Burgers().generate(
        FourierBasis, n, (space_modes, space_modes),
        u0="random", f="random", nu=0.05,
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.time_dependent and f.time_dependent
    assert u.coeff.shape[0] == n and not torch.isnan(u.coeff).any()


@pytest.mark.no_mms
@SETTINGS
@given(
    u0=st.floats(-1.0, 1.0),
    fval=st.floats(0.05, 0.5),
    space_modes=st.integers(8, 24),
    seed=st.integers(0, 10_000),
)
def test_burgers_numerical_constant_ic_and_forcing(u0, fval, space_modes, seed):
    # constant initial condition + constant forcing exercise the Number branches
    u, f = Burgers().generate(
        FourierBasis, 2, (space_modes, space_modes), u0=u0, f=fval, nu=0.05,
        space_domain=slice(0, 1, space_modes), time_domain=slice(0, 1, 40),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.time_dependent
    assert f.coeff.abs().max() > 0  # non-zero constant forcing
    assert not torch.isnan(u.coeff).any()


@pytest.mark.no_mms
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 24),
    nu=st.floats(0.01, 0.2),
    seed=st.integers(0, 10_000),
)
def test_burgers_numeric_residual_runs(n, space_modes, nu, seed):
    problem = Burgers()
    u, f = problem.generate(
        FourierBasis, n, (space_modes, space_modes), nu=nu, u0="random", f="random",
        generator=torch.Generator().manual_seed(seed),
    )
    res = problem.residual(u, f, nu, res=64)
    assert res.time_dependent
    assert not torch.isnan(res.coeff).any()


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 4), modes=st.integers(4, 24), nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_rhs_smoke(n, modes, nu, seed):
    # rhs is the spectral RK4 right-hand side kept for future numerical solvers
    g = torch.Generator().manual_seed(seed)
    u = FourierBasis.generate_coeff(n, modes, generator=g)
    f = torch.zeros(n, modes, dtype=torch.complex64)
    out = Burgers.rhs(FourierBasis, nu, u, f)
    assert out.shape == u.shape
    assert not torch.isnan(out).any()


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 3), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_antiderivative_complex_integration_constant(n, modes, seed):
    # complex u0 exercises the complex integration-constant branch
    u, ut = Antiderivative().generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed),
        u0=1 + 2j,
    )
    assert u.coeff.shape[0] == n
    assert torch.allclose(u.coeff[:, 0], torch.tensor(1 + 2j, dtype=u.coeff.dtype))


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 4), modes=st.integers(4, 16), seed=st.integers(0, 10_000))
def test_antiderivative_numeric_residual_runs(n, modes, seed):
    problem = Antiderivative()
    u, ut = problem.generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=0
    )
    res = problem.residual(u, ut)
    assert not torch.isnan(res.coeff).any()


@pytest.mark.no_mms
@SETTINGS
@given(u0=st.floats(-3.0, 3.0), n=st.integers(1, 3), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_antiderivative_float_integration_constant(u0, n, modes, seed):
    u, ut = Antiderivative().generate(
        FourierBasis, n, modes, generator=torch.Generator().manual_seed(seed), u0=u0
    )
    assert torch.allclose(
        u.coeff[:, 0], torch.tensor(u0 + 0j, dtype=u.coeff.dtype), atol=1e-5
    )


@pytest.mark.no_mms
@SETTINGS
@given(m=st.integers(8, 16), seed=st.integers(0, 10_000))
def test_burgers_int_modes_and_1d_ic(m, seed):
    # int modes -> (m, m); a 1D u0 tensor is broadcast across samples
    ic = torch.rand(m, dtype=torch.float64)
    u, _ = Burgers().generate(
        FourierBasis, 2, m, u0=ic, f=0, nu=0.05,
        space_domain=slice(0, 1, m), time_domain=slice(0, 1, 40),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u.coeff.shape[0] == 2 and u.coeff.shape[2] == m


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 3), m=st.integers(6, 12), nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_residual_non_time_dependent(n, m, nu, seed):
    # residual() on a non-time-dependent 2D field exercises the else branch
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.transform(torch.randn(n, m, m, generator=g) + 0j)
    u = FourierBasis(coeff, periods=(1.0, 1.0))
    f = FourierBasis(torch.zeros_like(coeff), periods=(1.0, 1.0))
    res = Burgers().residual(u, f, nu, res=m)
    assert not res.time_dependent
    assert not torch.isnan(res.coeff).any()


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 3), modes=st.integers(4, 10), seed=st.integers(0, 10_000))
def test_antiderivative_tensor_integration_constant(n, modes, seed):
    # a tensor u0 hits the final else branch that assigns it directly
    g = torch.Generator().manual_seed(seed)
    u0 = torch.randn(n, dtype=torch.complex64, generator=g)
    u, _ = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=u0)
    assert torch.allclose(u.coeff[:, 0], u0.to(u.coeff.dtype), atol=1e-5)
