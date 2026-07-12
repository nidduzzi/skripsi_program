"""Operator baselines: dataset, SpectralSVR adapter, benchmark, DeepONet smoke."""

import importlib.util

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.model import LSSVR
from SpectralSVR.operators import (
    DeepONetOperator,
    OperatorDataset,
    SpectralSVROperator,
    benchmark,
    sensor_grid,
)
from SpectralSVR.utils import get_metrics

_HAS_DEEPXDE = importlib.util.find_spec("deepxde") is not None
SETTINGS = settings(deadline=None, max_examples=5)


def _antiderivative_split(seed, n_train=120, n_test=30, modes=16):
    # operator maps the derivative ut -> the antiderivative u (integration)
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis, n_train + n_test, modes, generator=g, u0=0
    )
    train = OperatorDataset.from_fields(ut[:n_train], u[:n_train], n_sensors=modes)
    test = OperatorDataset.from_fields(ut[n_train:], u[n_train:], n_sensors=modes)
    return train, test


@pytest.mark.no_mms
@given(n=st.integers(2, 8), d=st.integers(1, 2))
def test_sensor_grid_shape(n, d):
    domain = tuple((0.0, 1.0) for _ in range(d))
    grid = sensor_grid(domain, n)
    assert grid.shape == (n**d, d)
    # half-open: endpoint excluded on every axis
    assert grid.max() < 1.0


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_operator_dataset_shapes():
    train, _ = _antiderivative_split(0)
    assert train.sensors.shape == (16, 1)
    assert train.f.shape == (120, 16)
    assert train.n_functions == 120
    x = torch.linspace(0, 1, 25).unsqueeze(-1)
    assert train.targets(x).shape == (120, 25)


@pytest.mark.no_mms
@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_spectralsvr_operator_learns_antiderivative(seed):
    train, test = _antiderivative_split(seed)
    op = SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0)).fit(train)
    query = torch.linspace(0, 1, 40).unsqueeze(-1)
    pred = op.predict(test.f, query).cpu()
    target = test.targets(query).cpu()
    assert pred.shape == target.shape
    assert get_metrics(pred, target)["r2"] > 0.5


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_predict_before_fit_raises():
    op = SpectralSVROperator(FourierBasis(), LSSVR())
    with pytest.raises(AssertionError, match="fit before predict"):
        op.predict(torch.randn(2, 16), torch.linspace(0, 1, 10).unsqueeze(-1))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_benchmark_returns_metric_table():
    train, test = _antiderivative_split(1)
    query = torch.linspace(0, 1, 40).unsqueeze(-1)
    df = benchmark(
        [SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0))],
        train,
        test,
        query,
    )
    assert list(df.index) == ["SpectralSVR"]
    for col in ("fit_seconds", "rmse", "r2", "rrse"):
        assert col in df.columns
    assert df.loc["SpectralSVR", "r2"] > 0.5


@pytest.mark.skipif(not _HAS_DEEPXDE, reason="needs the 'baselines' extra (deepxde)")
@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_deeponet_operator_smoke():
    train, test = _antiderivative_split(0, n_train=40, n_test=10)
    query = torch.linspace(0, 1, 20).unsqueeze(-1)
    op = DeepONetOperator(trunk=query, iterations=20, seed=0).fit(train)
    pred = op.predict(test.f, query)
    assert pred.shape == (test.n_functions, query.shape[0])
    assert not torch.isnan(pred).any()
