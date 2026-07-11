"""Fuzzed + MMS tests for the problem generators.

MMS here: a problem generates a solution ``u`` together with the forcing/rhs
that makes ``u`` an exact solution of the operator. We assert the generated
pair satisfies the defining equation (residual ~ 0).
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.problems import Burgers

SETTINGS = settings(
    deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow]
)


# --------------------------------------------------------------------------- #
# Antiderivative: du/dx == ut by construction (clean spectral MMS)
# --------------------------------------------------------------------------- #
@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(1, 5),
    modes=st.integers(3, 16),
    seed=st.integers(0, 10_000),
)
def test_antiderivative_spectral_residual_is_zero(n, modes, seed):
    gen = torch.Generator().manual_seed(seed)
    problem = Antiderivative()
    u, ut = problem.generate(FourierBasis, n, modes, generator=gen, u0=0)
    residual = problem.spectral_residual(u, ut)
    # u is manufactured so that ut == d/dx u exactly (up to the dropped DC mode)
    assert residual.coeff.abs().max() < 1e-3


@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(1, 4),
    modes=st.integers(4, 16),
    seed=st.integers(0, 10_000),
)
def test_antiderivative_grad_matches_generated_derivative(n, modes, seed):
    gen = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, n, modes, generator=gen, u0=0)
    grad = u.grad()
    # both have their DC mode zeroed by the spectral derivative
    assert torch.allclose(grad.coeff[:, 1:], ut.coeff[:, 1:], atol=1e-3)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_antiderivative_rejects_bad_sizes():
    problem = Antiderivative()
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 0, 8)
    with pytest.raises(AssertionError):
        problem.generate(FourierBasis, 4, 0)


# --------------------------------------------------------------------------- #
# Burgers: manufactured-solution generator
# --------------------------------------------------------------------------- #
@pytest.mark.no_mms
@SETTINGS
@given(
    n=st.integers(1, 3),
    space_modes=st.integers(8, 16),
    seed=st.integers(0, 10_000),
)
def test_burgers_generate_shapes_and_reproducible(n, space_modes, seed):
    problem = Burgers()
    kw = dict(nu=0.05, u0="random", f="random")
    u1, f1 = problem.generate(
        FourierBasis, n, (space_modes, space_modes),
        generator=torch.Generator().manual_seed(seed), **kw,
    )
    u2, f2 = problem.generate(
        FourierBasis, n, (space_modes, space_modes),
        generator=torch.Generator().manual_seed(seed), **kw,
    )
    assert u1.time_dependent and f1.time_dependent
    assert u1.coeff.shape == f1.coeff.shape
    assert u1.coeff.shape[0] == n and u1.coeff.shape[2] == space_modes
    assert torch.equal(u1.coeff, u2.coeff)  # reproducible under the same seed
    assert torch.equal(f1.coeff, f2.coeff)


@pytest.mark.mms
@pytest.mark.no_fuzz
@pytest.mark.xfail(
    reason="Burgers.generate resamples to nt=time_domain.step while modes[0] is "
    "small, so the manufactured (u, f) pair does not satisfy the equation; "
    "residual is currently ~O(solution). Tracked as a known consistency bug.",
    strict=True,
)
def test_burgers_manufactured_solution_satisfies_equation():
    problem = Burgers()
    nu = 0.05
    u, f = problem.generate(
        FourierBasis, 2, (16, 16), generator=torch.Generator().manual_seed(0),
        nu=nu, u0="random", f="random",
    )
    residual = problem.residual(u, f, nu, res=200)
    rel = residual.get_values(res=200).abs().mean() / u.get_values(res=200).abs().mean()
    assert rel < 1e-2
