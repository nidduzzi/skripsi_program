import torch

from SpectralSVR.model.FNN import FNN


def _tiny_trained_fnn():
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    y = torch.randn(16, 2)
    model = FNN(MAX_EPOCH=2, w_hidden=8, n_hidden=1, batch_size=8)
    model.fit(x, y)
    return model, x


def test_fnn_dump_load_roundtrip(tmp_path):
    model, x = _tiny_trained_fnn()
    preds_before = model.predict(x)

    path = str(tmp_path / "fnn_model")
    model.dump(path)
    loaded = FNN.load(path)

    assert loaded.trained
    preds_after = loaded.predict(x)
    assert torch.allclose(preds_before, preds_after, atol=1e-6)


def test_fnn_dump_only_hyperparams(tmp_path):
    model, _ = _tiny_trained_fnn()
    path = str(tmp_path / "fnn_hp")
    model.dump(path, only_hyperparams=True)
    loaded = FNN.load(path)

    assert not loaded.trained  # weights were not saved
    assert loaded.w_hidden == model.w_hidden
    assert loaded.n_hidden == model.n_hidden


def test_fnn_load_wrong_type_raises(tmp_path):
    path = str(tmp_path / "bad.pt")
    torch.save({"type": "LSSVR", "hyperparameters": {}}, path)
    try:
        FNN.load(path)
        raise AssertionError("expected type mismatch to raise")
    except Exception as e:
        assert "doesn't match 'FNN'" in str(e)
