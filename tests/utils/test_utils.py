import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from SpectralSVR.utils import (
    DEFAULT_METRICS,
    StandardScaler,
    euler_solver,
    get_metrics,
    resolve_device,
    scale_to_standard,
    to_complex_coeff,
    to_mag_angle,
    to_real_coeff,
)

SETTINGS = settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
_elements = st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False)


@st.composite
def real_arrays(draw):
    # to_complex_coeff pairs the last axis, so its width must be even.
    rows = draw(st.integers(1, 12))
    half = draw(st.integers(1, 16))
    return torch.from_numpy(
        draw(hnp.arrays(np.float64, (rows, 2 * half), elements=_elements))
    )


@pytest.mark.no_mms
@SETTINGS
@given(a=real_arrays())
def test_real_complex_real_roundtrip(a):
    # real -> complex -> real is a clean round trip for even width
    assert torch.allclose(to_real_coeff(to_complex_coeff(a)), a)


@pytest.mark.no_mms
@SETTINGS
@given(
    rows=st.integers(1, 12),
    cols=st.integers(1, 16),
    seed=st.integers(0, 10_000),
)
def test_complex_invertible(rows, cols, seed):
    c = torch.randn(rows, cols, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(to_complex_coeff(to_real_coeff(c)), c)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_resolve_device_explicit():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_resolve_device_default_matches_availability():
    expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert resolve_device(None) == expected


@pytest.mark.no_mms
@SETTINGS
@given(rows=st.integers(2, 20), cols=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_get_metrics_default_keys(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    preds = torch.randn(rows, cols, generator=g)
    targets = torch.randn(rows, cols, generator=g)
    m = get_metrics(preds, targets)
    assert set(m) == set(DEFAULT_METRICS) | {"pred_nan_sum"}
    assert m["mse"] >= 0.0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_metrics_single_sample_r2_is_nan():
    m = get_metrics(torch.randn(1, 3), torch.randn(1, 3))
    assert m["r2"] != m["r2"]  # NaN


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_get_metrics_injected_metric_set():
    m = get_metrics(
        torch.zeros(4, 2), torch.zeros(4, 2), metrics={"custom": lambda p, t: (p - t).abs().sum()}
    )
    assert m["custom"] == 0.0
    assert "mse" not in m
    assert "pred_nan_sum" in m


@pytest.mark.no_mms
# --------------------------------------------------------------------------- #
# StandardScaler
# --------------------------------------------------------------------------- #
@given(rows=st.integers(2, 40), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_roundtrip_real(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g) * 3 + 1.5
    scaler = StandardScaler().fit(x)
    assert torch.allclose(scaler.inverse(scaler.transform(x)), x, atol=1e-4)


@pytest.mark.no_mms
@given(rows=st.integers(2, 40), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_standardizes(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g) * 3 + 1.5
    xt = StandardScaler().fit(x).transform(x)
    assert torch.allclose(xt.mean(0), torch.zeros(cols), atol=1e-4)
    assert torch.allclose(xt.std(0, unbiased=False), torch.ones(cols), atol=1e-3)


@pytest.mark.no_mms
@given(rows=st.integers(2, 30), cols=st.integers(1, 6), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_roundtrip_complex(rows, cols, seed):
    c = torch.randn(rows, cols, dtype=torch.complex64,
                    generator=torch.Generator().manual_seed(seed))
    scaler = StandardScaler().fit(c)
    assert torch.allclose(scaler.inverse(scaler.transform(c)), c, atol=1e-3)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_standard_scaler_tuple_and_subset():
    a = torch.randn(10, 3)
    b = torch.randn(10, 2, dtype=torch.complex64)
    scaler = StandardScaler().fit((a, b))
    ta, tb = scaler.transform((a, b))
    assert ta.shape == a.shape and tb.shape == b.shape
    sub = scaler.get_subset_scaler(0)
    assert torch.allclose(sub.inverse(sub.transform(a)), a, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_standard_scaler_consistency_checks():
    scaler = StandardScaler().fit(torch.randn(8, 3))
    with pytest.raises(AssertionError):
        scaler.transform(torch.randn(8, 4))  # wrong 2nd dim
    with pytest.raises(AssertionError):
        scaler.transform(torch.randn(8, 3, dtype=torch.complex64))  # complex mismatch


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_standard_scaler_save_load(tmp_path):
    x = torch.randn(12, 4)
    scaler = StandardScaler().fit(x)
    path = str(tmp_path / "scaler.pt")
    scaler.save(path)
    loaded = StandardScaler.load(path)
    assert torch.allclose(loaded.transform(x), scaler.transform(x), atol=1e-6)


@pytest.mark.no_mms
@given(rows=st.integers(2, 30), cols=st.integers(1, 6), seed=st.integers(0, 10_000))
@SETTINGS
def test_scale_to_standard(rows, cols, seed):
    x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))
    xt = scale_to_standard(x)
    assert torch.allclose(xt.mean(0), torch.zeros(cols), atol=1e-4)


@pytest.mark.no_mms
# --------------------------------------------------------------------------- #
# to_mag_angle
# --------------------------------------------------------------------------- #
@given(rows=st.integers(1, 20), cols=st.integers(1, 10), seed=st.integers(0, 10_000))
@SETTINGS
def test_to_mag_angle_reconstructs(rows, cols, seed):
    c = torch.randn(rows, cols, dtype=torch.complex64,
                    generator=torch.Generator().manual_seed(seed))
    ma = to_mag_angle(c)
    assert ma.shape == (rows, 2 * cols)
    mag, angle = ma[:, ::2], ma[:, 1::2]
    assert torch.allclose(mag * torch.exp(1j * angle), c, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_to_mag_angle_idempotent_on_real():
    r = torch.randn(3, 4)
    assert torch.equal(to_mag_angle(r), r)


@pytest.mark.mms
# --------------------------------------------------------------------------- #
# euler_solver (MMS: dy/dt = -a y -> y0 exp(-a t))
# --------------------------------------------------------------------------- #
@given(a=st.floats(0.2, 3.0), seed=st.integers(0, 10_000))
@SETTINGS
def test_euler_solver_linear_decay(a, seed):
    y0 = torch.randn(2, generator=torch.Generator().manual_seed(seed))
    t = torch.linspace(0, 1.0, 2000)

    def rhs(ti, y):
        return -a * y

    sol = euler_solver(rhs, y0, t)
    exact = y0 * torch.exp(-a * t[-1])
    assert torch.allclose(sol[-1], exact, atol=5e-2 * y0.abs().max())


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_euler_solver_requires_multiple_points():
    with pytest.raises(AssertionError):
        euler_solver(lambda ti, y: y, torch.zeros(2), torch.tensor([0.0]))
