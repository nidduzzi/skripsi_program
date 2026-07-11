"""Tests for FourierBasis and the Basis base-class machinery it implements.

Correctness is cross-checked against PyTorch's own FFT (`torch.fft`) and
analytic derivatives/integrals wherever a ground truth exists.
"""

import math

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

import pytest  # noqa: E402
import torch  # noqa: E402

from SpectralSVR import FourierBasis, to_complex_coeff, to_real_coeff


def _make_signal(n_points: int = 64) -> torch.Tensor:
    """A smooth band-limited periodic signal, complex-cast, shape (1, n)."""
    x = torch.arange(0, 1, 1.0 / n_points)
    f = (
        3 * torch.sin(2 * torch.pi * 1.0 * x)
        + torch.sin(2 * torch.pi * 4.0 * x)
        + 0.5 * torch.cos(2 * torch.pi * 7.0 * x)
    )
    return (f + 0j).unsqueeze(0)


# --------------------------------------------------------------------------- #
# transform / inv_transform vs torch.fft
# --------------------------------------------------------------------------- #
def test_transform_matches_torch_fft():
    torch.manual_seed(0)
    f = torch.randn(3, 16) + 0j
    coeff = FourierBasis.transform(f, periodic=True)
    assert torch.allclose(coeff, torch.fft.fft(f, dim=1), atol=1e-4)


def test_inv_transform_matches_torch_ifft():
    torch.manual_seed(1)
    coeff = torch.randn(3, 16, dtype=torch.complex64)
    vals = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(vals, torch.fft.ifft(coeff, dim=1), atol=1e-4)


def test_transform_roundtrip():
    f = _make_signal(32)
    coeff = FourierBasis.transform(f, periodic=True)
    back = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(back, f, atol=1e-4)


def test_transform_2d_matches_torch_fft2():
    torch.manual_seed(2)
    f = torch.randn(2, 8, 8) + 0j
    coeff = FourierBasis.transform(f, periodic=True)
    assert torch.allclose(coeff, torch.fft.fft2(f, dim=(1, 2)), atol=1e-3)


def test_transform_2d_roundtrip():
    torch.manual_seed(3)
    f = torch.randn(2, 8, 8) + 0j
    coeff = FourierBasis.transform(f, periodic=True)
    back = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(back, f, atol=1e-3)


def test_transform_non_fft_path_matches_fft_path():
    # allow_fft=False takes the explicit matmul branch; must agree with FFT.
    torch.manual_seed(4)
    f = torch.randn(2, 12) + 0j
    fft_path = FourierBasis.transform(f, periodic=True, allow_fft=True)
    mm_path = FourierBasis.transform(f, periodic=True, allow_fft=False)
    assert torch.allclose(fft_path, mm_path, atol=1e-3)


def test_transform_requires_2d():
    with pytest.raises(AssertionError):
        FourierBasis.transform(torch.randn(8) + 0j)


def test_transform_casts_real_to_complex():
    f = torch.randn(1, 8)  # real input, not complex
    coeff = FourierBasis.transform(f, periodic=True)
    assert coeff.is_complex()


# --------------------------------------------------------------------------- #
# wave_number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modes", [1, 2, 5, 6, 7, 16])
def test_wave_number_matches_fftfreq(modes):
    k = FourierBasis.wave_number(modes).flatten()
    expected = torch.fft.fftfreq(modes) * modes
    assert torch.allclose(k, expected)


# --------------------------------------------------------------------------- #
# fn (basis function values)
# --------------------------------------------------------------------------- #
def test_fn_shape_and_values():
    x = torch.linspace(0, 1, 5).view(-1, 1)
    basis = FourierBasis.fn(x, modes=4)
    assert basis.shape == (5, 4)
    # column k is exp(2j pi k x)
    k = FourierBasis.wave_number(4).flatten()
    expected = torch.exp(2j * torch.pi * x * k)
    assert torch.allclose(basis, expected, atol=1e-5)


def test_fn_requires_modes():
    with pytest.raises(ValueError):
        FourierBasis.fn(torch.zeros(2, 1))


def test_fn_dimension_mismatch():
    with pytest.raises(AssertionError):
        FourierBasis.fn(torch.zeros(2, 1), modes=(4, 4))  # x is 1D, modes 2D


# --------------------------------------------------------------------------- #
# grad / integral vs analytic
# --------------------------------------------------------------------------- #
def test_grad_matches_analytic_derivative():
    n = 64
    x = torch.arange(0, 1, 1.0 / n)
    sig = (torch.sin(2 * torch.pi * x) + 0j).unsqueeze(0)
    basis = FourierBasis(FourierBasis.transform(sig, periodic=True))
    g = basis.grad()
    dval = g.inv_transform(g.coeff, periodic=True).real.flatten()
    analytic = 2 * torch.pi * torch.cos(2 * torch.pi * x)
    assert torch.allclose(dval, analytic, atol=1e-2)


def test_grad_multiplier_matches_coeff_formula():
    # grad coeff == coeff * 2j*pi*k / period (mode 0 zeroed)
    coeff = FourierBasis.transform(_make_signal(32), periodic=True)
    basis = FourierBasis(coeff)
    k = FourierBasis.wave_number(basis.modes[0])
    expected = coeff * 2j * torch.pi * k.T
    expected[:, 0] = 0
    assert torch.allclose(basis.grad().coeff, expected, atol=1e-4)


def test_grad_order_two():
    n = 64
    x = torch.arange(0, 1, 1.0 / n)
    sig = (torch.sin(2 * torch.pi * x) + 0j).unsqueeze(0)
    basis = FourierBasis(FourierBasis.transform(sig, periodic=True))
    g2 = basis.grad(ord=2)
    dval = g2.inv_transform(g2.coeff, periodic=True).real.flatten()
    analytic = -((2 * torch.pi) ** 2) * torch.sin(2 * torch.pi * x)
    assert torch.allclose(dval, analytic, atol=1e-1)


def test_integral_inverts_grad_for_zero_mean():
    # For a zero-mean signal, integral(grad(f)) recovers f (mode 0 is dropped).
    sig = _make_signal(32)  # sum of sinusoids -> zero mean
    coeff = FourierBasis.transform(sig, periodic=True)
    basis = FourierBasis(coeff)
    recovered = basis.grad().integral().coeff
    expected = coeff.clone()
    expected[:, 0] = 0
    assert torch.allclose(recovered, expected, atol=1e-3)


# --------------------------------------------------------------------------- #
# evaluate / __call__
# --------------------------------------------------------------------------- #
def test_call_interpolates_between_samples():
    # Fit on coarse grid, evaluate on a finer/shifted grid, compare to truth.
    n = 100
    t_fit = torch.arange(0, 1, 1.0 / n)
    f = (
        3 * torch.sin(2 * torch.pi * 1.0 * t_fit)
        + torch.sin(2 * torch.pi * 4.0 * t_fit)
        + 0.5 * torch.sin(2 * torch.pi * 7.0 * t_fit)
    ) + 0j
    basis = FourierBasis(FourierBasis.transform(f.unsqueeze(0), periodic=True))

    t_eval = torch.arange(-1, 1, 1.0 / 150)
    truth = (
        3 * torch.sin(2 * torch.pi * 1.0 * t_eval)
        + torch.sin(2 * torch.pi * 4.0 * t_eval)
        + 0.5 * torch.sin(2 * torch.pi * 7.0 * t_eval)
    )
    pred = basis(t_eval)[0].real
    assert pred.shape == truth.shape
    assert torch.norm(pred - truth, 2) < 1e-2


def test_call_requires_coeff():
    basis = FourierBasis()  # empty coeff
    with pytest.raises(AssertionError):
        basis(torch.linspace(0, 1, 10))


def test_evaluate_i_n_selection():
    coeff = FourierBasis.transform(torch.randn(4, 16) + 0j, periodic=True)
    x = torch.linspace(0, 1, 20)
    out = FourierBasis.evaluate(coeff, x, i=1, n=2)
    assert out.shape[0] == 2


def test_evaluate_accepts_real_coeff():
    coeff = FourierBasis.transform(torch.randn(1, 16) + 0j, periodic=True)
    real = to_real_coeff(coeff)
    x = torch.linspace(0, 1, 20)
    out_c = FourierBasis.evaluate(coeff, x)
    out_r = FourierBasis.evaluate(real, x)
    assert torch.allclose(out_c, out_r, atol=1e-4)


# --------------------------------------------------------------------------- #
# generate / generate_coeff / generate_empty
# --------------------------------------------------------------------------- #
def test_generate_empty_shape_and_zero():
    coeff = FourierBasis.generate_empty(3, (8, 4))
    assert coeff.shape == (3, 8, 4)
    assert coeff.is_complex()
    assert torch.count_nonzero(coeff) == 0


def test_generate_coeff_shape_and_dtype():
    coeff = FourierBasis.generate_coeff(5, 8)
    assert coeff.shape == (5, 8)
    assert coeff.is_complex()


def test_generate_coeff_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = FourierBasis.generate_coeff(4, 8, generator=g1)
    b = FourierBasis.generate_coeff(4, 8, generator=g2)
    assert torch.equal(a, b)


def test_generate_real_signal_has_conjugate_symmetry():
    # non-complex_funcs generation should yield real-valued functions
    coeff = FourierBasis.generate_coeff(1, 16, complex_funcs=False)
    vals = FourierBasis.inv_transform(coeff, periodic=True)
    assert torch.allclose(vals.imag, torch.zeros_like(vals.imag), atol=1e-4)


def test_generate_returns_basis():
    g = torch.Generator().manual_seed(0)
    basis = FourierBasis.generate(2, 8, generator=g)
    assert isinstance(basis, FourierBasis)
    assert basis.modes == (8,)
    assert len(basis) == 2


def test_generate_zero_value_type():
    basis = FourierBasis.generate(2, 8, value_type="zero")
    assert torch.count_nonzero(basis.coeff) == 0


def test_generate_invalid_value_type():
    with pytest.raises(RuntimeError):
        FourierBasis.generate(2, 8, value_type="bogus")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# coeff setter / properties
# --------------------------------------------------------------------------- #
def test_coeff_setter_rejects_non_complex():
    with pytest.raises(AssertionError):
        FourierBasis(torch.randn(1, 8))  # real coeff


def test_empty_coeff_defaults():
    basis = FourierBasis()
    assert basis.coeff.numel() == 0
    assert len(basis) == 0


def test_modes_and_ndim_1d():
    basis = FourierBasis(FourierBasis.generate_empty(2, 8))
    assert basis.modes == (8,)
    assert basis.ndim == 1


def test_modes_and_ndim_2d():
    basis = FourierBasis(FourierBasis.generate_empty(2, (8, 4)))
    assert basis.modes == (8, 4)
    assert basis.ndim == 2


def test_periods_default_and_set():
    basis = FourierBasis(FourierBasis.generate_empty(1, (8, 4)), periods=(2.0, 3.0))
    assert basis.periods == (2.0, 3.0)


def test_len_matches_first_dim():
    basis = FourierBasis(FourierBasis.generate_empty(7, 8))
    assert len(basis) == 7


# --------------------------------------------------------------------------- #
# resize_modes / perturb / indexing / arithmetic
# --------------------------------------------------------------------------- #
def test_resize_modes_up_and_down():
    basis = FourierBasis(FourierBasis.generate_empty(1, 16))
    up = basis.resize_modes(32)
    assert up.modes[0] == 32
    down = basis.resize_modes(8)
    assert down.modes[0] == 8


def test_resize_modes_preserves_function_values():
    # Zero-padding in frequency space (rescale=False) upsamples without
    # changing the underlying band-limited function.
    sig = _make_signal(32)
    coeff = FourierBasis.transform(sig, periodic=True)
    basis = FourierBasis(coeff)
    up = basis.resize_modes(64, rescale=False)
    original = basis.get_values(res=64)[0].real.flatten()
    upsampled = up.get_values(res=64)[0].real.flatten()
    assert torch.allclose(upsampled, original, atol=1e-2)


def test_perturb_changes_values_and_keeps_shape():
    g = torch.Generator().manual_seed(0)
    basis = FourierBasis(FourierBasis.generate_coeff(1, 32, generator=g))
    perturbed = basis.perturb(std_ratio=0.1, generator=g)
    assert perturbed.coeff.shape == basis.coeff.shape
    assert not torch.allclose(perturbed.coeff, basis.coeff)


def test_getitem_selects_functions():
    basis = FourierBasis(FourierBasis.generate_empty(5, 8))
    sub = basis[:2]
    assert len(sub) == 2


def test_add_and_sub():
    coeff = FourierBasis.transform(_make_signal(16), periodic=True)
    a = FourierBasis(coeff)
    doubled = a + a
    assert torch.allclose(doubled.coeff, coeff * 2, atol=1e-4)
    zero = a - a
    assert torch.allclose(zero.coeff, torch.zeros_like(coeff), atol=1e-4)


def test_add_type_error():
    a = FourierBasis(FourierBasis.generate_empty(1, 8))
    with pytest.raises(TypeError):
        _ = a + 5  # type: ignore[operator]


def test_copy_is_independent():
    coeff = FourierBasis.transform(_make_signal(16), periodic=True)
    a = FourierBasis(coeff)
    b = a.copy()
    b.coeff = b.coeff * 2
    assert not torch.allclose(a.coeff, b.coeff)


# --------------------------------------------------------------------------- #
# get_values / get_values_and_grid / grid
# --------------------------------------------------------------------------- #
def test_get_values_and_grid_shapes():
    basis = FourierBasis(FourierBasis.transform(_make_signal(16), periodic=True))
    vals, grid = basis.get_values_and_grid(res=50)
    assert vals.shape[0] == 1
    assert grid.shape[-1] == 1


def test_grid_1d():
    grid = FourierBasis.grid(slice(0, 1, 10))
    assert grid.shape == (10, 1)
    assert torch.isclose(grid[0, 0], torch.tensor(0.0))
    assert torch.isclose(grid[-1, 0], torch.tensor(1.0))


def test_grid_2d():
    grid = FourierBasis.grid((slice(0, 1, 4), slice(0, 2, 6)))
    assert grid.shape == (4, 6, 2)


def test_prefered_evaluation_mode():
    assert FourierBasis.prefered_evaluation_mode() == "inverse transform"


# --------------------------------------------------------------------------- #
# time dependent conversion
# --------------------------------------------------------------------------- #
def test_to_time_dependent_and_back():
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(2, 8, 8) + 0j, periodic=True)
    basis = FourierBasis(coeff)
    td = basis.to_time_dependent()
    assert td.time_dependent
    assert td.time_size > 0
    ti = td.to_time_independent()
    assert not ti.time_dependent


# --------------------------------------------------------------------------- #
# to_real_coeff / to_complex_coeff round trip
# --------------------------------------------------------------------------- #
def test_real_complex_coeff_roundtrip():
    coeff = FourierBasis.transform(_make_signal(16), periodic=True)
    real = to_real_coeff(coeff)
    assert not real.is_complex()
    back = to_complex_coeff(real)
    assert torch.equal(back, coeff)


def test_to_complex_coeff_idempotent_on_complex():
    coeff = FourierBasis.transform(_make_signal(16), periodic=True)
    assert torch.equal(to_complex_coeff(coeff), coeff)


def test_to_real_coeff_idempotent_on_real():
    real = torch.randn(2, 8)
    assert torch.equal(to_real_coeff(real), real)


# --------------------------------------------------------------------------- #
# periods affect derivative scaling
# --------------------------------------------------------------------------- #
def test_grad_respects_period():
    n = 64
    period = 2.0
    x = torch.arange(0, period, period / n)
    sig = (torch.sin(2 * torch.pi * x / period) + 0j).unsqueeze(0)
    coeff = FourierBasis.transform(sig, periodic=True, periods=period)
    basis = FourierBasis(coeff, periods=period)
    g = basis.grad()
    dval = g.inv_transform(g.coeff, periodic=True, periods=period).real.flatten()
    analytic = (2 * torch.pi / period) * torch.cos(2 * torch.pi * x / period)
    assert torch.allclose(dval, analytic, atol=1e-2)


CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# basis-evaluation mode (regression: res_t was unbound for non-time-dependent)
# --------------------------------------------------------------------------- #
def test_get_values_basis_eval_mode_1d():
    basis = FourierBasis(FourierBasis.transform(_make_signal(16), periodic=True))
    vals = basis.get_values(res=16, evaluation_mode="basis", device=CPU)
    assert vals.shape == (1, 16)


def test_get_values_basis_eval_mode_2d():
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    basis = FourierBasis(coeff)
    vals = basis.get_values(res=8, evaluation_mode="basis", device=CPU)
    assert vals.shape[0] == 1
    assert vals.ndim == 3


# --------------------------------------------------------------------------- #
# time-dependent grad / integral (finite-difference branch)
# --------------------------------------------------------------------------- #
def _time_dependent_basis():
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(2, 8, 8) + 0j, periodic=True)
    return FourierBasis(coeff).to_time_dependent(nt=8)


def test_time_dependent_grad_time_dim():
    td = _time_dependent_basis()
    g = td.grad(dim=0)  # finite-difference in time
    assert g.coeff.shape == td.coeff.shape
    assert g.time_dependent


def test_time_dependent_integral_time_dim():
    td = _time_dependent_basis()
    ig = td.integral(dim=0)  # cumulative-sum in time
    assert ig.coeff.shape == td.coeff.shape


def test_time_dependent_grad_spatial_dim():
    td = _time_dependent_basis()
    gs = td.grad(dim=1)  # spectral derivative in space
    assert gs.coeff.shape == td.coeff.shape


# --------------------------------------------------------------------------- #
# complex_funcs generation path
# --------------------------------------------------------------------------- #
def test_generate_complex_funcs_true():
    g = torch.Generator().manual_seed(1)
    basis = FourierBasis.generate(1, 16, complex_funcs=True, generator=g)
    assert basis.coeff.is_complex()
    # complex functions generally carry a nonzero imaginary part
    vals = FourierBasis.inv_transform(basis.coeff, periodic=True)
    assert vals.imag.abs().sum() > 0


# --------------------------------------------------------------------------- #
# plot (headless Agg smoke tests across the branches)
# --------------------------------------------------------------------------- #
def _real_1d():
    sig = _make_signal(32)
    return FourierBasis(FourierBasis.transform(sig, periodic=True))


def _complex_1d():
    coeff = FourierBasis.generate_coeff(1, 32, complex_funcs=True)
    return FourierBasis(coeff, complex_funcs=True)


def test_plot_1d_real():
    _real_1d().plot(device=CPU)


def test_plot_1d_complex_both_components():
    _complex_1d().plot(device=CPU)


@pytest.mark.parametrize("component", ["real", "imag"])
def test_plot_1d_complex_single_component(component):
    _complex_1d().plot(plot_component=component, device=CPU)


def test_plot_1d_complex_scatter():
    _complex_1d().plot(complex_scatter=True, device=CPU)


def test_plot_2d():
    torch.manual_seed(0)
    coeff = FourierBasis.transform(torch.randn(1, 8, 8) + 0j, periodic=True)
    FourierBasis(coeff).plot(device=CPU)


def test_plot_out_of_bounds():
    basis = FourierBasis(FourierBasis.generate_empty(1, 8))
    with pytest.raises(AssertionError):
        basis.plot(i=0, n=5, device=CPU)


def test_math_import_smoke():
    # guard: ensure module-level torch pi usage stays consistent
    assert math.isclose(float(torch.pi), math.pi, rel_tol=1e-6)
