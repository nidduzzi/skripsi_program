"""Fuzzed + MMS + serialization tests for FNN."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from SpectralSVR.model.FNN import FNN

from _common import SETTINGS, constant_dataset, linear_dataset


@pytest.mark.mms
@SETTINGS
@given(const=st.floats(-3.0, 3.0), seed=st.integers(0, 10_000))
def test_fnn_overfits_constant_target(const, seed):
    # Manufactured constant target; a small FNN trained long enough recovers it.
    X, y = constant_dataset(32, 3, 2, const, seed)
    model = FNN(MAX_EPOCH=400, w_hidden=16, n_hidden=2, batch_size=32, lr=0.01)
    model.fit(X, y)
    assert torch.allclose(model.predict(X), y, atol=1e-1)


@pytest.mark.no_mms
@SETTINGS
@given(
    n=st.integers(8, 24),
    d=st.integers(1, 4),
    o=st.integers(1, 3),
    seed=st.integers(0, 10_000),
)
def test_fnn_predict_shape(n, d, o, seed):
    X, y = linear_dataset(n, d, o, seed)
    model = FNN(MAX_EPOCH=2, w_hidden=8, n_hidden=1, batch_size=8)
    model.fit(X, y)
    assert model.predict(X).shape == (n, o)


def _tiny_trained_fnn():
    X, y = linear_dataset(16, 3, 2, seed=0)
    model = FNN(MAX_EPOCH=2, w_hidden=8, n_hidden=1, batch_size=8)
    model.fit(X, y)
    return model, X


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fnn_dump_load_roundtrip(tmp_path):
    model, x = _tiny_trained_fnn()
    preds_before = model.predict(x)

    path = str(tmp_path / "fnn_model")
    model.dump(path)
    loaded = FNN.load(path)

    assert loaded.trained
    assert torch.allclose(preds_before, loaded.predict(x), atol=1e-6)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fnn_dump_only_hyperparams(tmp_path):
    model, _ = _tiny_trained_fnn()
    path = str(tmp_path / "fnn_hp")
    model.dump(path, only_hyperparams=True)
    loaded = FNN.load(path)

    assert not loaded.trained  # weights were not saved
    assert loaded.w_hidden == model.w_hidden
    assert loaded.n_hidden == model.n_hidden


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_fnn_load_wrong_type_raises(tmp_path):
    path = str(tmp_path / "bad.pt")
    torch.save({"type": "LSSVR", "hyperparameters": {}}, path)
    with pytest.raises(Exception, match="doesn't match 'FNN'"):
        FNN.load(path)
