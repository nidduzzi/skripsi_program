"""Fuzzed + MMS tests for LSSVR."""

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR import LSSVR

from _model_data import SETTINGS, linear_dataset, smooth_dataset


@pytest.mark.mms
@SETTINGS
@given(
    n=st.integers(30, 60),
    d=st.integers(2, 5),
    o=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_lssvr_linear_recovers_linear_map(n, d, o, seed):
    # Manufacture y = X W + b; a linear-kernel LSSVR recovers it. The Gram
    # system is rank-deficient/ill-conditioned; C=1e4 regularizes it and float64
    # keeps the lstsq stable on adversarial random draws (float32 drifts).
    X, y = linear_dataset(n, d, o, seed)
    model = LSSVR(kernel="linear", C=1e4, dtype=torch.float64)
    model.fit(X.double(), y.double())
    assert torch.allclose(model.predict(X.double()), y.double(), atol=1e-4)


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


@pytest.mark.no_mms
@SETTINGS
@given(
    kernel=st.sampled_from(["linear", "poly", "rbf", "tri", "frob", "max"]),
    n=st.integers(10, 30),
    seed=st.integers(0, 10_000),
)
def test_lssvr_all_kernels_fit_predict(kernel, n, seed):
    # every kernel builds, fits and predicts a finite output of the right shape
    X, y = linear_dataset(n, 3, 2, seed)
    model = LSSVR(kernel=kernel, sigma=1.5, d=2.0)
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape == y.shape
    assert not torch.isnan(out).any()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_invalid_kernel_raises():
    with pytest.raises(KeyError):
        LSSVR(kernel="bogus").fit(torch.randn(6, 2), torch.randn(6, 1))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_correlation_and_p_matrix():
    X, y = linear_dataset(20, 3, 2, seed=0)
    model = LSSVR(kernel="linear", C=10.0)
    model.fit(X, y)
    corr = model.get_correlation_image()
    pmat = model.get_p_matrix()
    assert corr.shape[0] == X.shape[0]
    assert pmat.shape == (X.shape[1], y.shape[1])


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_correlation_before_fit_raises():
    with pytest.raises(AssertionError):
        LSSVR().get_correlation_image()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_dump_only_hyperparams(tmp_path):
    X, y = linear_dataset(20, 3, 2, seed=0)
    model = LSSVR(kernel="rbf", C=7.0)
    model.fit(X, y)
    path = str(tmp_path / "lssvr_hp")
    model.dump(path, only_hyperparams=True)
    loaded = LSSVR.load(path, only_hyperparams=True)
    assert loaded.C == 7.0
    assert not loaded.trained  # parameters not saved


from SpectralSVR.model import torch_json_encoder  # noqa: E402


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_kernel_property_before_fit_defaults_sigma():
    # accessing K before fit (sv_x is None) falls back to sigma=1.0
    model = LSSVR(kernel="rbf")
    assert callable(model.K)
    assert model.kernel_params["sigma"] == 1.0


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(6, 40), d=st.integers(1, 4), seed=st.integers(0, 10_000))
def test_lssvr_batched_kernel_matches_unbatched(n, d, seed):
    # tiny batch_size forces the chunked kernel loop; result must match the
    # single-shot kernel exactly.
    X, y = linear_dataset(n, d, 2, seed)
    unbatched = LSSVR(kernel="rbf", C=10.0, dtype=torch.float64)
    unbatched.fit(X.double(), y.double())
    batched = LSSVR(
        kernel="rbf", C=10.0, dtype=torch.float64, batch_size_func=lambda dims: 1
    )
    batched.fit(X.double(), y.double())
    assert torch.allclose(batched.predict(X.double()), unbatched.predict(X.double()), atol=1e-8)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_load_type_mismatch_raises(tmp_path):
    import json

    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"type": "FNN", "hyperparameters": {}}))
    with pytest.raises(Exception, match="doesn't match 'LSSVR'"):
        LSSVR.load(str(path))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_torch_json_encoder_rejects_unknown_type():
    with pytest.raises(TypeError):
        torch_json_encoder({1, 2, 3})  # not a torch object


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(6, 30), d=st.integers(1, 4), seed=st.integers(0, 10_000))
def test_lssvr_fit_accepts_numpy_and_1d_targets(n, d, seed):
    # numpy inputs go through from_numpy; 1D y is viewed as a column
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)  # 1D targets
    model = LSSVR(kernel="linear")
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape[0] == n


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_lssvr_device_setter():
    model = LSSVR()
    model.device = torch.device("cpu")
    assert model.device == torch.device("cpu")


@pytest.mark.no_fuzz
@pytest.mark.no_mms
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "tri", "frob", "max"])
def test_lssvr_each_kernel_covered(kernel):
    # deterministic: every kernel function is exercised on every run
    X, y = linear_dataset(15, 3, 2, seed=0)
    model = LSSVR(kernel=kernel, sigma=1.5, d=2.0)
    model.fit(X, y)
    out = model.predict(X)
    assert out.shape == y.shape and not torch.isnan(out).any()
