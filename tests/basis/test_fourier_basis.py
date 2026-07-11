"""Tests for FourierBasis and the Basis base-class machinery it implements.

Testing strategy (enforced by ``conftest.py``):

* Every test is either fuzzed with Hypothesis (``@given``) or explicitly
  opts out with ``@pytest.mark.no_fuzz``.
* Every test either uses the Method of Manufactured Solutions
  (``@pytest.mark.mms``) or opts out with ``@pytest.mark.no_mms``.

MMS here means: we *choose* an exact spectrum ``c``, manufacture the
function ``u = inv_transform(c)`` from it, and then assert that the code
recovers the known analytic answer (the spectrum itself, its spectral
derivative/integral, or the closed-form point evaluation).

Correctness is additionally cross-checked against PyTorch's own FFT.
"""

import math

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from SpectralSVR import (  # noqa: E402
    FourierBasis,
    to_complex_coeff,
    to_real_coeff,
)

CPU = torch.device("cpu")

# Hypothesis is slow with torch; drop the per-example deadline and allow the
# fixed-shape function-scoped helpers.
SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #
_elements = st.floats(
    min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False
)


@st.composite
def spectra_1d(draw, min_n=2, max_n=32, max_rows=3):
    """A manufactured complex spectrum of shape (rows, n)."""
    n = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    re = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    im = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    return torch.from_numpy(re + 1j * im).to(torch.complex128)


@st.composite
def real_signals_1d(draw, min_n=2, max_n=32, max_rows=3):
    n = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    a = draw(hnp.arrays(np.float64, (rows, n), elements=_elements))
    return torch.from_numpy(a).to(torch.complex128)


@st.composite
def spectra_2d(draw, min_n=2, max_n=10, max_rows=2):
    r = draw(st.integers(min_n, max_n))
    c = draw(st.integers(min_n, max_n))
    rows = draw(st.integers(1, max_rows))
    re = draw(hnp.arrays(np.float64, (rows, r, c), elements=_elements))
    im = draw(hnp.arrays(np.float64, (rows, r, c), elements=_elements))
    return torch.from_numpy(re + 1j * im).to(torch.complex128)


# =========================================================================== #
# MMS tests: manufacture a solution from a known spectrum, recover the answer
# =========================================================================== #
@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_transform_recovers_spectrum(coeff):
    # Manufacture u from the known spectrum, then the forward transform must
    # return that exact spectrum.
    u = FourierBasis.inv_transform(coeff, periodic=True)
    recovered = FourierBasis.transform(u, periodic=True)
    assert torch.allclose(recovered, coeff, atol=1e-6)
    # torch.fft agrees on both legs of the manufacture/recover round trip.
    assert torch.allclose(u, torch.fft.ifft(coeff, dim=1), atol=1e-6)
    assert torch.allclose(recovered, torch.fft.fft(u, dim=1), atol=1e-6)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_2d())
def test_mms_transform_recovers_spectrum_2d(coeff):
    u = FourierBasis.inv_transform(coeff, periodic=True)
    recovered = FourierBasis.transform(u, periodic=True)
    assert torch.allclose(recovered, coeff, atol=1e-5)
    # torch.fft 2D agrees on the manufacture and recovery.
    assert torch.allclose(u, torch.fft.ifft2(coeff, dim=(1, 2)), atol=1e-5)
    assert torch.allclose(recovered, torch.fft.fft2(u, dim=(1, 2)), atol=1e-5)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_grad_is_spectral_derivative(coeff):
    # Known spectrum => known derivative spectrum: c_k * 2*pi*i*k (period 1),
    # with the DC (k=0) mode dropped.
    n = coeff.shape[1]
    basis = FourierBasis(coeff)
    k = FourierBasis.wave_number(n).T.to(coeff)
    expected = coeff * 2j * torch.pi * k
    expected[:, 0] = 0
    grad_coeff = basis.grad().coeff
    assert torch.allclose(grad_coeff, expected, atol=1e-6)
    # torch.fft-based spectral derivative in value space agrees.
    k_torch = (torch.fft.fftfreq(n) * n).to(coeff)
    u = FourierBasis.inv_transform(coeff, periodic=True)
    dval_torch = torch.fft.ifft(torch.fft.fft(u, dim=1) * 2j * torch.pi * k_torch, dim=1)
    dval_ours = basis.grad().inv_transform(grad_coeff, periodic=True)
    assert torch.allclose(dval_ours, dval_torch, atol=1e-4, rtol=1e-6)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_integral_inverts_grad(coeff):
    # grad then integral recovers the manufactured spectrum, minus its DC mode.
    coeff = coeff.clone()
    coeff[:, 0] = 0  # zero-mean manufactured solution
    basis = FourierBasis(coeff)
    recovered = basis.grad().integral().coeff
    assert torch.allclose(recovered, coeff, atol=1e-6)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d(max_n=12, max_rows=2))
def test_mms_evaluate_closed_form(coeff):
    # Point evaluation must equal the closed form (1/N) sum_k c_k exp(2pi i k x).
    n = coeff.shape[1]
    x = torch.tensor([[0.1], [0.37], [0.83]], dtype=torch.float64)
    out = FourierBasis.evaluate(coeff, x)
    k = FourierBasis.wave_number(n).flatten().to(coeff)
    closed = (coeff.unsqueeze(1) * torch.exp(2j * torch.pi * x * k)).sum(-1) / n
    assert torch.allclose(out, closed, atol=1e-6)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_evaluate_matches_inv_transform_on_grid(coeff):
    # Evaluating the manufactured function on the sample grid reproduces the
    # inverse transform used to manufacture it.
    n = coeff.shape[1]
    basis = FourierBasis(coeff)
    grid = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    ev = basis(grid, device=CPU)
    iv = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(ev, iv, atol=1e-6)
    # both match torch.fft.ifft of the manufactured spectrum on the grid.
    assert torch.allclose(ev, torch.fft.ifft(coeff, dim=1), atol=1e-6)


@pytest.mark.mms
@SETTINGS
@given(coeff=spectra_1d())
def test_mms_parseval(coeff):
    # Parseval for this DFT convention: sum|u|^2 == (1/N) sum|c|^2.
    n = coeff.shape[1]
    u = FourierBasis.inv_transform(coeff, periodic=True)
    lhs = (u.abs() ** 2).sum()
    rhs = (coeff.abs() ** 2).sum() / n
    assert torch.isclose(lhs, rhs, atol=1e-5, rtol=1e-5)


@pytest.mark.mms
@SETTINGS
@given(
    amp=st.floats(0.5, 5.0),
    freq=st.integers(1, 5),
    period=st.floats(0.5, 4.0),
)
def test_mms_derivative_of_sine_with_period(amp, freq, period):
    # Manufactured u = A sin(2*pi*m*x/L); u' = A(2*pi*m/L) cos(2*pi*m*x/L).
    n = 64
    x = torch.arange(0, period, period / n, dtype=torch.float64)
    u = (amp * torch.sin(2 * torch.pi * freq * x / period) + 0j).unsqueeze(0)
    coeff = FourierBasis.transform(u, periodic=True, periods=period)
    basis = FourierBasis(coeff, periods=period)
    g = basis.grad()
    dval = g.inv_transform(g.coeff, periodic=True, periods=period).real.flatten()
    analytic = amp * (2 * torch.pi * freq / period) * torch.cos(
        2 * torch.pi * freq * x / period
    )
    assert torch.allclose(dval, analytic, atol=1e-2)


@pytest.mark.mms
@SETTINGS
@given(amp=st.floats(0.5, 5.0), freq=st.integers(1, 6))
def test_mms_integral_of_cosine(amp, freq):
    # Manufactured u = A cos(2*pi*m*x); integral = A sin(2*pi*m*x)/(2*pi*m).
    n = 128
    x = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    u = (amp * torch.cos(2 * torch.pi * freq * x) + 0j).unsqueeze(0)
    basis = FourierBasis(FourierBasis.transform(u, periodic=True))
    ig = basis.integral()
    val = ig.inv_transform(ig.coeff, periodic=True).real.flatten()
    analytic = amp * torch.sin(2 * torch.pi * freq * x) / (2 * torch.pi * freq)
    assert torch.allclose(val, analytic, atol=1e-2)


@pytest.mark.mms
@SETTINGS
@given(amp=st.floats(-5.0, 5.0))
def test_mms_constant_is_dc_only(amp):
    # Manufactured constant A -> spectrum is A*N at DC, zero elsewhere.
    n = 16
    f = torch.full((1, n), amp, dtype=torch.float64) + 0j
    coeff = FourierBasis.transform(f, periodic=True)
    assert torch.isclose(
        coeff[0, 0].real, torch.tensor(amp * n, dtype=torch.float64), atol=1e-4
    )
    assert coeff[0, 1:].abs().max() < 1e-4


@pytest.mark.mms
@SETTINGS
@given(freq=st.integers(1, 7), amp=st.floats(0.5, 4.0))
def test_mms_cosine_spectrum(freq, amp):
    # A cos(2*pi*m*x) -> real peaks A*N/2 at +m and -m (== N-m), zero elsewhere.
    n = 16
    assert 0 < freq < n // 2
    x = torch.arange(0, 1, 1.0 / n, dtype=torch.float64)
    coeff = FourierBasis.transform(
        (amp * torch.cos(2 * torch.pi * freq * x) + 0j).unsqueeze(0), periodic=True
    )
    peak = torch.tensor(amp * n / 2, dtype=torch.float64)
    assert torch.isclose(coeff[0, freq].real, peak, atol=1e-3)
    assert torch.isclose(coeff[0, n - freq].real, peak, atol=1e-3)


# =========================================================================== #
# Fuzzed cross-checks against torch.fft and structural invariants (no MMS)
# =========================================================================== #
@pytest.mark.no_mms
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_transform_matches_torch_fft(f):
    assert torch.allclose(
        FourierBasis.transform(f, periodic=True), torch.fft.fft(f, dim=1), atol=1e-6
    )


@pytest.mark.no_mms
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_inv_transform_matches_torch_ifft(coeff):
    assert torch.allclose(
        FourierBasis.inv_transform(coeff, periodic=True),
        torch.fft.ifft(coeff, dim=1),
        atol=1e-6,
    )


@pytest.mark.no_mms
@SETTINGS
@given(f=spectra_2d())
def test_fuzz_transform_matches_torch_fft2(f):
    assert torch.allclose(
        FourierBasis.transform(f, periodic=True),
        torch.fft.fft2(f, dim=(1, 2)),
        atol=1e-5,
    )


@pytest.mark.no_mms
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_roundtrip_real(f):
    coeff = FourierBasis.transform(f, periodic=True)
    back = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(back, f, atol=1e-6)


@pytest.mark.no_mms
@SETTINGS
@given(f=real_signals_1d())
def test_fuzz_transform_non_fft_path_matches_fft_path(f):
    fft_path = FourierBasis.transform(f, periodic=True, allow_fft=True)
    mm_path = FourierBasis.transform(f, periodic=True, allow_fft=False)
    # The explicit matmul path accumulates more float error than the FFT.
    assert torch.allclose(fft_path, mm_path, atol=1e-3, rtol=1e-4)


@pytest.mark.no_mms
@SETTINGS
@given(a=st.floats(-3, 3), b=st.floats(-3, 3), f=spectra_1d(), g=spectra_1d())
def test_fuzz_transform_linearity(a, b, f, g):
    n = min(f.shape[1], g.shape[1])
    rows = min(f.shape[0], g.shape[0])
    f, g = f[:rows, :n], g[:rows, :n]
    lhs = FourierBasis.transform(a * f + b * g, periodic=True)
    rhs = a * FourierBasis.transform(f, periodic=True) + b * FourierBasis.transform(
        g, periodic=True
    )
    assert torch.allclose(lhs, rhs, atol=1e-5)


@pytest.mark.no_mms
@SETTINGS
@given(modes=st.integers(1, 64))
def test_fuzz_wave_number_matches_fftfreq(modes):
    k = FourierBasis.wave_number(modes).flatten()
    assert torch.allclose(k, torch.fft.fftfreq(modes) * modes)


@pytest.mark.no_mms
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_real_complex_coeff_roundtrip(coeff):
    c64 = coeff.to(torch.complex64)
    assert torch.equal(to_complex_coeff(to_real_coeff(c64)), c64)


@pytest.mark.no_mms
@SETTINGS
@given(
    points=st.integers(2, 20),
    modes=st.integers(1, 16),
)
def test_fuzz_fn_columns_are_complex_exponentials(points, modes):
    x = torch.linspace(0, 1, points, dtype=torch.float64).view(-1, 1)
    basis = FourierBasis.fn(x, modes=modes)
    k = FourierBasis.wave_number(modes).flatten().to(x)
    expected = torch.exp(2j * torch.pi * x * k)
    assert basis.shape == (points, modes)
    assert torch.allclose(basis, expected, atol=1e-6)


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 5), modes=st.integers(2, 16), seed=st.integers(0, 10_000))
def test_fuzz_generate_coeff_reproducible(n, modes, seed):
    a = FourierBasis.generate_coeff(n, modes, generator=torch.Generator().manual_seed(seed))
    b = FourierBasis.generate_coeff(n, modes, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(a, b)
    assert a.shape == (n, modes)


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(1, 4), modes=st.integers(4, 32), seed=st.integers(0, 10_000))
def test_fuzz_generate_real_signal_has_no_imaginary_part(n, modes, seed):
    coeff = FourierBasis.generate_coeff(
        n, modes, complex_funcs=False, generator=torch.Generator().manual_seed(seed)
    )
    vals = FourierBasis.inv_transform(coeff, periodic=True)
    assert vals.imag.abs().max() < 1e-4


@pytest.mark.no_mms
@SETTINGS
@given(target=st.integers(2, 64))
def test_fuzz_resize_modes_sets_target(target):
    basis = FourierBasis(FourierBasis.generate_empty(2, 16))
    resized = basis.resize_modes(target)
    assert resized.modes[0] == target


@pytest.mark.no_mms
@SETTINGS
@given(seed=st.integers(0, 10_000), std=st.floats(0.01, 0.5))
def test_fuzz_perturb_keeps_shape_and_changes_values(seed, std):
    gen = torch.Generator().manual_seed(seed)
    basis = FourierBasis(FourierBasis.generate_coeff(1, 32, generator=gen))
    perturbed = basis.perturb(std_ratio=std, generator=gen)
    assert perturbed.coeff.shape == basis.coeff.shape
    assert not torch.allclose(perturbed.coeff, basis.coeff)


@pytest.mark.no_mms
@SETTINGS
@given(coeff=spectra_1d())
def test_fuzz_add_doubles_and_sub_zeroes(coeff):
    basis = FourierBasis(coeff)
    assert torch.allclose((basis + basis).coeff, coeff * 2, atol=1e-6)
    assert torch.allclose((basis - basis).coeff, torch.zeros_like(coeff), atol=1e-6)


@pytest.mark.no_mms
@SETTINGS
@given(rows=st.integers(2, 6), stop=st.integers(1, 2))
def test_fuzz_getitem_selects_rows(rows, stop):
    stop = min(stop, rows)
    basis = FourierBasis(FourierBasis.generate_empty(rows, 8))
    assert len(basis[:stop]) == stop


# =========================================================================== #
# Structural / error-path tests: explicitly not fuzzed, no MMS
# =========================================================================== #
@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_transform_requires_2d():
    with pytest.raises(AssertionError):
        FourierBasis.transform(torch.randn(8) + 0j)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_inv_transform_requires_2d():
    with pytest.raises(AssertionError):
        FourierBasis.inv_transform(torch.randn(8) + 0j)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_transform_casts_real_to_complex():
    assert FourierBasis.transform(torch.randn(1, 8), periodic=True).is_complex()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_inv_transform_casts_real_input():
    assert FourierBasis.inv_transform(torch.randn(1, 8)).is_complex()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_transform_res_int_and_slice():
    f = torch.randn(1, 8) + 0j
    assert FourierBasis.transform(f, res=8, periodic=True).shape == (1, 8)
    assert FourierBasis.transform(f, res=slice(0, 1, 8), periodic=True).shape == (1, 8)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fn_requires_modes():
    with pytest.raises(ValueError):
        FourierBasis.fn(torch.zeros(2, 1))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fn_dimension_mismatch():
    with pytest.raises(AssertionError):
        FourierBasis.fn(torch.zeros(2, 1), modes=(4, 4))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fn_casts_integer_input():
    basis = FourierBasis.fn(torch.arange(4).view(-1, 1), modes=4)
    assert basis.shape == (4, 4) and basis.is_complex()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fn_transpose_matches_shape():
    x = torch.linspace(0, 1, 4).view(-1, 1)
    assert FourierBasis.fn(x, 4, transpose=True).shape == FourierBasis.fn(x, 4).shape


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_coeff_setter_rejects_non_complex():
    with pytest.raises(AssertionError):
        FourierBasis(torch.randn(1, 8))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_empty_basis_modes_and_len():
    basis = FourierBasis()
    assert basis.coeff.numel() == 0
    assert basis.modes == (0,)
    assert len(basis) == 0
    assert basis.ndim == 0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_modes_ndim_periods_time_size():
    basis = FourierBasis(FourierBasis.generate_empty(2, (8, 4)), periods=(2.0, 3.0))
    assert basis.modes == (8, 4)
    assert basis.ndim == 2
    assert basis.periods == (2.0, 3.0)
    assert basis.time_size == 0  # not time dependent


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_len_matches_first_dim():
    assert len(FourierBasis(FourierBasis.generate_empty(7, 8))) == 7


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_call_requires_coeff():
    with pytest.raises(AssertionError):
        FourierBasis()(torch.linspace(0, 1, 10))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_evaluate_accepts_real_coeff():
    coeff = FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True)
    x = torch.linspace(0, 1, 20)
    out_c = FourierBasis.evaluate(coeff, x)
    out_r = FourierBasis.evaluate(to_real_coeff(coeff), x)  # real coeff path
    assert torch.allclose(out_c, out_r, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_res_slice():
    basis = FourierBasis(FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True))
    vals = basis.get_values(res=slice(0, 1, 20), device=CPU)
    assert vals.shape[0] == 1


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_res_tuple():
    basis = FourierBasis(FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True))
    vals = basis.get_values(res=(slice(0, 1, 20),), device=CPU)
    assert vals.shape[0] == 1


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_evaluate_i_n_selection():
    coeff = FourierBasis.transform(torch.randn(4, 16) + 0j, periodic=True)
    out = FourierBasis.evaluate(coeff, torch.linspace(0, 1, 20), i=1, n=2)
    assert out.shape[0] == 2


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_evaluate_time_dependent():
    coeff = FourierBasis.transform(torch.randn(2, 8, 8) + 0j, periodic=True)
    out = FourierBasis.evaluate(
        coeff, torch.linspace(0, 1, 10).view(-1, 1),
        t=torch.linspace(0, 1, 5), time_dependent=True, periods=(1.0, 1.0),
    )
    assert out.shape == (2, 5, 10)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_evaluate_time_dependent_requires_t():
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    with pytest.raises(AssertionError):
        FourierBasis.evaluate(
            coeff, torch.linspace(0, 1, 5).view(-1, 1),
            time_dependent=True, periods=(1.0, 1.0),
        )


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_generate_returns_basis():
    basis = FourierBasis.generate(2, 8, generator=torch.Generator().manual_seed(0))
    assert isinstance(basis, FourierBasis) and basis.modes == (8,) and len(basis) == 2


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_generate_zero_value_type():
    assert torch.count_nonzero(FourierBasis.generate(2, 8, value_type="zero").coeff) == 0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_generate_empty_shape():
    coeff = FourierBasis.generate_empty(3, (8, 4))
    assert coeff.shape == (3, 8, 4) and coeff.is_complex()
    assert torch.count_nonzero(coeff) == 0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_generate_complex_funcs_true():
    basis = FourierBasis.generate(
        1, 16, complex_funcs=True, generator=torch.Generator().manual_seed(1)
    )
    assert basis.coeff.is_complex()
    vals = FourierBasis.inv_transform(basis.coeff, periodic=True)
    assert vals.imag.abs().sum() > 0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_generate_invalid_value_type():
    with pytest.raises(RuntimeError):
        FourierBasis.generate(2, 8, value_type="bogus")  # type: ignore[arg-type]


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_add_type_error():
    with pytest.raises(TypeError):
        _ = FourierBasis(FourierBasis.generate_empty(1, 8)) + 5  # type: ignore[operator]


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_sub_type_error():
    with pytest.raises(TypeError):
        _ = FourierBasis(FourierBasis.generate_empty(1, 8)) - 5  # type: ignore[operator]


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_copy_is_independent():
    coeff = FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True)
    a = FourierBasis(coeff)
    b = a.copy()
    b.coeff = b.coeff * 2
    assert not torch.allclose(a.coeff, b.coeff)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_prefered_evaluation_mode():
    assert FourierBasis.prefered_evaluation_mode() == "inverse transform"


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_grid_1d_and_2d():
    g1 = FourierBasis.grid(slice(0, 1, 10))
    assert g1.shape == (10, 1)
    assert torch.isclose(g1[0, 0], torch.tensor(0.0))
    assert torch.isclose(g1[-1, 0], torch.tensor(1.0))
    assert FourierBasis.grid((slice(0, 1, 4), slice(0, 2, 6))).shape == (4, 6, 2)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_grid_int_arg():
    assert FourierBasis.grid(10).shape == (10, 1)


# --------------------------------------------------------------------------- #
# get_values / evaluation modes (inverse-transform and basis-eval)
# --------------------------------------------------------------------------- #
@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_and_grid_shapes():
    basis = FourierBasis(FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True))
    vals, grid = basis.get_values_and_grid(res=50, device=CPU)
    assert vals.shape[0] == 1 and grid.shape[-1] == 1


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_basis_eval_1d():
    basis = FourierBasis(FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True))
    assert basis.get_values(res=16, evaluation_mode="basis", device=CPU).shape == (1, 16)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_basis_eval_2d():
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    vals = FourierBasis(coeff).get_values(res=8, evaluation_mode="basis", device=CPU)
    assert vals.ndim == 3 and vals.shape[0] == 1


# --------------------------------------------------------------------------- #
# time-dependent conversion, grad, integral, resize
# --------------------------------------------------------------------------- #
def _time_dependent_basis(modes=8, nt=8):
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(2, modes, modes) + 0j, periodic=True)
    return FourierBasis(coeff).to_time_dependent(nt=nt)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_to_time_dependent_and_back():
    td = _time_dependent_basis()
    assert td.time_dependent and td.time_size == 8
    ti = td.to_time_independent()
    assert not ti.time_dependent


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_to_time_dependent_default_nt():
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    td = FourierBasis(coeff).to_time_dependent()  # nt defaults to modes[0]
    assert td.time_dependent and td.time_size == 8


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_to_time_dependent_idempotent_when_already_time_dependent():
    td = _time_dependent_basis()
    again = td.to_time_dependent()  # early-return branch
    assert again.time_dependent


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_to_time_independent_noop_when_not_time_dependent():
    basis = FourierBasis(FourierBasis.transform(torch.randn(1, 8) + 0j, periodic=True))
    out = basis.to_time_independent()  # early-return branch
    assert not out.time_dependent


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_time_dependent_grad_time_and_space():
    td = _time_dependent_basis()
    assert td.grad(dim=0).coeff.shape == td.coeff.shape  # finite-diff in time
    assert td.grad(dim=1).coeff.shape == td.coeff.shape  # spectral in space


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_time_dependent_integral_time_and_space():
    td = _time_dependent_basis()
    assert td.integral(dim=0).coeff.shape == td.coeff.shape
    assert td.integral(dim=1).coeff.shape == td.coeff.shape


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_values_time_dependent_inverse_and_basis():
    td = _time_dependent_basis()
    assert td.get_values(device=CPU).shape == (2, 8, 8)
    basis_eval = td.get_values(
        res=(slice(0, 1, 8), slice(0, 1, 8)), evaluation_mode="basis", device=CPU
    )
    assert basis_eval.shape[0] == 2


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_resize_time_dependent_to_basis_interpolates_time():
    td = _time_dependent_basis(modes=8, nt=8)
    other = FourierBasis(
        FourierBasis.transform(torch.randn(1, 4, 4) + 0j, periodic=True)
    ).to_time_dependent(nt=4)
    resized = td.resize_modes(other)
    assert resized.modes == (4,)
    assert resized.time_size == 4


# --------------------------------------------------------------------------- #
# plot (headless Agg smoke tests across every branch)
# --------------------------------------------------------------------------- #
def _real_1d():
    x = torch.arange(0, 1, 1.0 / 32)
    sig = (torch.sin(2 * torch.pi * x) + 0j).unsqueeze(0)
    return FourierBasis(FourierBasis.transform(sig, periodic=True))


def _complex_1d():
    return FourierBasis(FourierBasis.generate_coeff(1, 32, complex_funcs=True), complex_funcs=True)


def _complex_2d():
    return FourierBasis(
        FourierBasis.generate_coeff(1, (8, 8), complex_funcs=True), complex_funcs=True
    )


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_1d_real():
    _real_1d().plot(device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_1d_complex_both_components():
    _complex_1d().plot(device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
@pytest.mark.parametrize("component", ["real", "imag"])
def test_plot_1d_complex_single_component(component):
    _complex_1d().plot(plot_component=component, device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_1d_complex_scatter():
    _complex_1d().plot(complex_scatter=True, device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_2d_real():
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    FourierBasis(coeff).plot(device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_2d_complex_default_component_warns():
    # complex 2D with no component defaults to real and logs a warning.
    _complex_2d().plot(device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
@pytest.mark.parametrize("component", ["real", "imag"])
def test_plot_2d_component_on_axes(component):
    _fig, ax = plt.subplots()
    _complex_2d().plot(plot_component=component, plt=ax, device=CPU)
    plt.close(_fig)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_2d_complex_scatter():
    _complex_2d().plot(complex_scatter=True, device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_plot_out_of_bounds():
    with pytest.raises(AssertionError):
        FourierBasis(FourierBasis.generate_empty(1, 8)).plot(i=0, n=5, device=CPU)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_math_import_smoke():
    assert math.isclose(float(torch.pi), math.pi, rel_tol=1e-6)
