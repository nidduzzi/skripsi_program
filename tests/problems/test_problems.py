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
    burgers_periodic_manufactured,
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
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))

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


# --------------------------------------------------------------------------- #
# Residual-operator validation against INDEPENDENT exact fields.
#
# The MMS/generate tests above are self-referential for the operator: they feed
# a pair the operator itself built, so residual == 0 only proves determinism.
# These feed fields + forcings derived from ANALYTIC derivatives (never from the
# operator under test), so they pin down the operator's actual coefficients and
# signs -- and also assert the non-trivial output (residual of a non-solution
# equals the known analytic residual, so an always-zero operator would fail).
# --------------------------------------------------------------------------- #
@pytest.mark.mms
@SETTINGS
@given(m=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_antiderivative_spectral_residual_matches_analytic(m, seed):
    problem = Antiderivative()
    length, modes = 1.0, 48
    amp = float(0.5 + torch.rand(1, generator=torch.Generator().manual_seed(seed)))
    x = space_grid(modes, length)
    k = 2 * math.pi * m / length
    u_val = (amp * torch.sin(k * x)).view(1, -1)
    ut_val = (amp * k * torch.cos(k * x)).view(1, -1)  # analytic derivative
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))
    ut = FourierBasis(FourierBasis.transform(ut_val + 0j), domain=(0.0, length))
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=(0.0, length))

    # true pair (ut from analytic cos, not u.grad) -> residual vanishes
    r0 = problem.spectral_residual(u, ut)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=1e-4)
    # non-solution (ut = 0) -> residual equals the analytic derivative
    r = problem.spectral_residual(u, zero)
    assert torch.allclose(r.inv_transform(r.coeff).real, ut_val, atol=1e-4)


@pytest.mark.mms
@SETTINGS
@given(m=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_antiderivative_finite_diff_residual_matches_analytic(m, seed):
    # residual() differentiates with torch.gradient, which is not periodic-aware,
    # so it carries O(1) error at the wrap boundary; validate the interior.
    problem = Antiderivative()
    length, modes, b = 1.0, 96, 8  # fine grid; b = boundary margin to exclude
    amp = float(0.5 + torch.rand(1, generator=torch.Generator().manual_seed(seed)))
    x = space_grid(modes, length)
    k = 2 * math.pi * m / length
    u_val = (amp * torch.sin(k * x)).view(1, -1)
    ut_val = (amp * k * torch.cos(k * x)).view(1, -1)
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=(0.0, length))
    ut = FourierBasis(FourierBasis.transform(ut_val + 0j), domain=(0.0, length))

    # true pair -> interior residual vanishes. This alone pins the operator: a
    # wrong derivative term leaves finite_diff(u) - ut != 0 (ut is independent).
    # The non-solution check lives in the spectral test -- here residual() would
    # transform the non-periodic finite difference, spreading boundary Gibbs into
    # the interior and making a value-space reconstruction unreliable.
    tol = 0.05 * float(ut_val.abs().max())  # finite-difference accuracy
    r0 = problem.residual(u, ut)
    r0v = r0.inv_transform(r0.coeff).real[:, b:-b]
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=tol)


@pytest.mark.mms
@SETTINGS
@given(nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_spectral_residual_matches_analytic(nu, seed):
    # space-time periodic manufactured field; forcing from analytic derivatives
    problem = Burgers()
    nt = ns = 16
    u_val, f_val = burgers_periodic_manufactured(nt, ns, nu)
    domain = ((0.0, 1.0), (0.0, 1.0))
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=domain)
    f = FourierBasis(FourierBasis.transform(f_val + 0j), domain=domain)
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=domain)

    # exact forcing -> residual vanishes to spectral accuracy
    r0 = problem.spectral_residual(u, f, nu)
    r0v = r0.inv_transform(r0.coeff).real
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=1e-4)
    # zero forcing -> residual equals the analytic Burgers operator applied to u
    r = problem.spectral_residual(u, zero, nu)
    assert torch.allclose(r.inv_transform(r.coeff).real, f_val, atol=1e-4)


@pytest.mark.mms
@SETTINGS
@given(nu=st.floats(0.01, 0.2), seed=st.integers(0, 10_000))
def test_burgers_finite_diff_residual_matches_analytic(nu, seed):
    # finite-diff residual: not periodic-aware, so validate the interior only.
    problem = Burgers()
    nt = ns = 16
    res, b = 96, 8  # fine grid; boundary margin
    u_val, f_val = burgers_periodic_manufactured(nt, ns, nu)
    domain = ((0.0, 1.0), (0.0, 1.0))
    u = FourierBasis(FourierBasis.transform(u_val + 0j), domain=domain)
    f = FourierBasis(FourierBasis.transform(f_val + 0j), domain=domain)
    zero = FourierBasis(torch.zeros_like(u.coeff), domain=domain)

    f_on_grid = f.get_values(res=res).real
    tol = 0.05 * float(f_on_grid.abs().max())  # finite-difference accuracy
    r0 = problem.residual(u, f, nu, res=res)
    r0v = r0.inv_transform(r0.coeff).real[:, b:-b, b:-b]
    assert torch.allclose(r0v, torch.zeros_like(r0v), atol=tol)
    rn = problem.residual(u, zero, nu, res=res)  # equals the operator applied to u
    rnv = rn.inv_transform(rn.coeff).real
    assert torch.allclose(rnv[:, b:-b, b:-b], f_on_grid[:, b:-b, b:-b], atol=tol)


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
    t_start=st.floats(-3.0, 3.0),
    t_len=st.floats(0.5, 3.0),
    x_start=st.floats(-3.0, 3.0),
    x_len=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
def test_burgers_mms_solution_satisfies_equation(
    n, space_modes, nu, t_start, t_len, x_start, x_len, seed
):
    # Fuzzing the per-axis (start, length) -- starts span 0 -- exercises MMS
    # exactness on both the canonical [0, L) and shifted/negative-start domains.
    # grad works off axis lengths, so a shift keeps the residual at machine zero.
    problem = Burgers()
    modes = (space_modes, space_modes)
    domain = ((t_start, t_start + t_len), (x_start, x_start + x_len))
    u, f = problem.mms_solution(
        FourierBasis, n, modes, nu=nu, domain=domain,
        generator=torch.Generator().manual_seed(seed),
    )
    residual = problem.spectral_residual(u, f, nu)
    assert residual.coeff.abs().max() < 1e-4
    assert f.coeff.abs().max() > 0  # non-trivial forcing
    assert u.domain == domain


@pytest.mark.no_mms
@SETTINGS
@given(
    x_start=st.floats(-3.0, 3.0),
    x_len=st.floats(0.5, 3.0),
    t_start=st.floats(-3.0, 3.0),
    t_len=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
def test_burgers_generate_honors_domain_start(x_start, x_len, t_start, t_len, seed):
    # generate() must thread space_domain.start / time_domain.start into the
    # output field's domain, not just the lengths.
    problem = Burgers()
    u_gen, _ = problem.generate(
        FourierBasis, 2, (8, 8),
        space_domain=slice(x_start, x_start + x_len, 64),
        time_domain=slice(t_start, t_start + t_len, 16),
        generator=torch.Generator().manual_seed(seed),
    )
    assert u_gen.domain[0] == (t_start, t_start + t_len)  # time axis start honored
    assert u_gen.domain[1] == (x_start, x_start + x_len)  # space axis start honored


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
    u = FourierBasis(coeff, domain=((0.0, 1.0), (0.0, 1.0)))
    f = FourierBasis(torch.zeros_like(coeff), domain=((0.0, 1.0), (0.0, 1.0)))
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
