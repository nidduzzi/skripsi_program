import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from SpectralSVR.utils import (
    DEFAULT_METRICS,
    get_metrics,
    resolve_device,
    to_complex_coeff,
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
