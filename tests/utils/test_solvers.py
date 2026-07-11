"""Fuzzed + MMS tests for the ETDRK4 exponential integrator."""

import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from SpectralSVR.utils import etdrk4_solver

SETTINGS = settings(
    deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.mark.mms
@SETTINGS
@given(
    N=st.integers(8, 64),
    nu=st.floats(0.01, 0.5),
    seed=st.integers(0, 10_000),
)
def test_etdrk4_linear_heat_matches_exact_1d(N, nu, seed):
    # With N(v)=0 the ODE dv/dt = L v has the exact solution v(t)=v0 exp(L t);
    # ETDRK4 integrates the linear part exactly, so it must match to fp error.
    k = torch.fft.fftfreq(N) * N
    linear = -nu * k**2
    v0 = torch.randn(3, N, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed))
    t = torch.linspace(0, 1.0, 50, dtype=torch.float64)

    def zero_nl(ti, v):
        return torch.zeros_like(v)

    sol = etdrk4_solver(linear, zero_nl, v0, t)
    exact = v0 * torch.exp(linear * float(t[-1]))
    assert torch.allclose(sol[-1], exact, atol=1e-8)


@pytest.mark.mms
@SETTINGS
@given(N=st.integers(4, 16), nu=st.floats(0.01, 0.3), seed=st.integers(0, 10_000))
def test_etdrk4_linear_heat_matches_exact_2d(N, nu, seed):
    # Same exact-integration check on a 2D mode grid (dimension-agnostic solver).
    k = torch.fft.fftfreq(N) * N
    kx = k.view(-1, 1)
    ky = k.view(1, -1)
    linear = -nu * (kx**2 + ky**2)
    v0 = torch.randn(2, N, N, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed))
    t = torch.linspace(0, 0.5, 30, dtype=torch.float64)

    def zero_nl(ti, v):
        return torch.zeros_like(v)

    sol = etdrk4_solver(linear, zero_nl, v0, t)
    exact = v0 * torch.exp(linear * float(t[-1]))
    assert torch.allclose(sol[-1], exact, atol=1e-8)


@pytest.mark.no_mms
@SETTINGS
@given(N=st.sampled_from([64, 128]), seed=st.integers(0, 10_000))
def test_etdrk4_burgers_is_stable(N, seed):
    # Nonlinear viscous Burgers should integrate without blowing up and should
    # not increase energy (viscous dissipation).
    length = 2 * math.pi
    k = torch.fft.fftfreq(N, d=length / N) * 2 * math.pi
    nu = 0.05
    linear = -nu * k**2
    dealias = (torch.fft.fftfreq(N) * N).abs() <= N // 3

    def burgers_nl(ti, v):
        u = torch.fft.ifft(v, dim=-1).real.to(v.dtype)
        return -0.5j * k * torch.fft.fft(u * u, dim=-1) * dealias

    x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1]
    u0 = torch.sin(x).unsqueeze(0) + 0.1 * torch.randn(
        2, N, generator=torch.Generator().manual_seed(seed)
    )
    v0 = torch.fft.fft(u0 + 0j, dim=-1)
    t = torch.linspace(0, 1.0, 100, dtype=torch.float64)

    sol = etdrk4_solver(linear, burgers_nl, v0, t)
    u = torch.fft.ifft(sol, dim=-1).real
    assert not torch.isnan(u).any()
    e0 = (u[0] ** 2).mean()
    ef = (u[-1] ** 2).mean()
    assert ef <= e0 + 1e-6


@pytest.mark.mms
@SETTINGS
@given(
    mode=st.integers(1, 4),
    nu=st.floats(0.02, 0.2),
    seed=st.integers(0, 10_000),
)
def test_etdrk4_reproduces_forced_manufactured_solution(mode, nu, seed):
    # MMS fuzzing the nonlinear solver: manufacture u = A sin(kx+phi) exp(-nu k^2 t),
    # which solves the heat equation. Forcing f = u u_x makes it an exact Burgers
    # solution, so integrating u_t = -u u_x + nu u_xx + f must reproduce u.
    N = 128
    length = 2 * math.pi
    g = torch.Generator().manual_seed(seed)
    k = (2 * math.pi * torch.fft.fftfreq(N, d=length / N)).view(1, N)
    kx = 2 * math.pi * mode / length
    A = (0.5 + torch.rand(2, 1, generator=g)).double()
    phi = (2 * math.pi * torch.rand(2, 1, generator=g)).double()
    x = torch.linspace(0, length, N + 1, dtype=torch.float64)[:-1].view(1, N)
    dealias = (torch.fft.fftfreq(N) * N).abs() <= N // 3

    def u_exact(tt):
        return A * torch.sin(kx * x + phi) * math.exp(-nu * kx**2 * tt)

    def ux_exact(tt):
        return A * kx * torch.cos(kx * x + phi) * math.exp(-nu * kx**2 * tt)

    def nonlinear(ti, v):
        u = torch.fft.ifft(v, dim=-1).real.to(v.dtype)
        advection = -0.5j * k * torch.fft.fft(u * u, dim=-1) * dealias
        forcing = torch.fft.fft((u_exact(float(ti)) * ux_exact(float(ti))).to(v.dtype), dim=-1)
        return advection + forcing

    linear = -nu * k.flatten() ** 2
    v0 = torch.fft.fft(u_exact(0.0) + 0j, dim=-1)
    t = torch.linspace(0, 1.0, 300, dtype=torch.float64)
    sol = etdrk4_solver(linear, nonlinear, v0, t)
    u_final = torch.fft.ifft(sol[-1], dim=-1).real
    exact = u_exact(1.0)
    assert (u_final - exact).abs().max() / exact.abs().max() < 1e-2


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_etdrk4_requires_matching_shapes():
    with pytest.raises(AssertionError):
        etdrk4_solver(
            torch.zeros(8),
            lambda ti, v: v,
            torch.zeros(2, 4, dtype=torch.complex128),
            torch.linspace(0, 1, 5),
        )
