"""Fuzzed + MMS tests for LSSVR."""

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR import LSSVR

from _common import SETTINGS, linear_dataset, smooth_dataset


@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(30, 60),
    d=st.integers(2, 5),
    o=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_lssvr_linear_recovers_linear_map(n, d, o, seed):
    # Manufacture y = X W + b; a linear-kernel LSSVR recovers it. C=1e4 keeps
    # the (rank-deficient) Gram system well-conditioned in float32; a larger C
    # leaves it near-singular and the float32 lstsq becomes unstable.
    X, y = linear_dataset(n, d, o, seed)
    model = LSSVR(kernel="linear", C=1e4)
    model.fit(X, y)
    assert torch.allclose(model.predict(X), y, atol=1e-2)


@pytest.mark.mms
@SETTINGS
@given(n=st.integers(20, 50), seed=st.integers(0, 10_000))
def test_lssvr_rbf_interpolates_training_points(n, seed):
    # Manufactured smooth target; rbf LSSVR with large C interpolates closely.
    X, y = smooth_dataset(n, seed)
    model = LSSVR(kernel="rbf", C=1e3)
    model.fit(X, y)
    assert torch.allclose(model.predict(X), y, atol=3e-1)


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(10, 40), d=st.integers(1, 4), o=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_lssvr_predict_numpy_roundtrips(n, d, o, seed):
    X, y = linear_dataset(n, d, o, seed)
    model = LSSVR(kernel="linear")
    model.fit(X, y)
    out = model.predict(X.numpy())
    assert isinstance(out, np.ndarray)
    assert out.shape == (n, o)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_dump_load_roundtrip(tmp_path):
    X, y = linear_dataset(30, 3, 2, seed=0)
    model = LSSVR(kernel="linear", C=10.0)
    model.fit(X, y)
    preds = model.predict(X)

    path = str(tmp_path / "lssvr")
    model.dump(path)
    loaded = LSSVR.load(path)
    assert loaded.trained
    assert torch.allclose(loaded.predict(X), preds, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_predict_before_fit_raises():
    with pytest.raises(AssertionError):
        LSSVR().predict(torch.randn(4, 2))
