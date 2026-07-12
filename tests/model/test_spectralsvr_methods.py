"""Coverage for SpectralSVR.test / inverse / inverse_coeff and error paths.

Behavioral tests are fuzzed over sizes/seed; error paths are structural and
opt out of fuzzing. None manufacture a PDE solution, so all are no_mms.
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from SpectralSVR import (
    Antiderivative,
    FourierBasis,
    LSSVR,
    SpectralSVR,
    to_real_coeff,
)

SETTINGS = settings(
    deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow]
)


def _trained_model(n=80, modes=8, seed=42, kernel="rbf"):
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=0)
    f = to_real_coeff(ut.coeff)  # real input features
    model = SpectralSVR(FourierBasis(periods=[1.0]), LSSVR(kernel=kernel, C=10.0))
    model.train(f, u.coeff)
    return model, f, u


# --------------------------------------------------------------------------- #
# behavioral (fuzzed)
# --------------------------------------------------------------------------- #
@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_test_returns_metric_groups(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    metrics = model.test(f, u.coeff)
    assert set(metrics) == {"spectral", "function value"}
    for group in metrics.values():
        assert "mse" in group and group["mse"] >= 0.0


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_inverse_coeff_shape(n, modes, seed):
    model, _, u = _trained_model(n, modes, seed)
    out = model.inverse_coeff(u.coeff, epochs=3)
    assert out.shape == (u.coeff.shape[0], modes)
    assert out.is_complex()


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_inverse_returns_field_values(n, modes, seed):
    model, _, u = _trained_model(n, modes, seed)
    points = FourierBasis.grid(slice(0, 1, 20))
    f_pred = model.inverse(u.coeff, points, epochs=3)
    assert f_pred.shape[0] == u.coeff.shape[0]
    assert not torch.isnan(f_pred).any()


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_forward_1d_x_and_complex_input(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    x = torch.linspace(0, 1, 20)  # 1D coordinate vector -> unsqueezed internally
    assert model.forward(f, x).shape[0] == n
    # complex input features are converted to interleaved real internally
    cf = torch.complex(f[:, ::2], f[:, 1::2])
    assert model.forward(cf, x).shape[0] == n


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_test_with_complex_input(n, modes, seed):
    model, f, u = _trained_model(n, modes, seed)
    cf = torch.complex(f[:, ::2], f[:, 1::2])  # complex features
    assert "spectral" in model.test(cf, u.coeff)


@pytest.mark.no_mms
@SETTINGS
@given(
    n=st.integers(20, 60),
    nt=st.integers(2, 8),
    modes=st.integers(4, 10),
    seed=st.integers(0, 10_000),
)
def test_test_time_dependent_and_complex_funcs(n, nt, modes, seed):
    # time-dependent targets + complex-valued functions exercise those branches
    g = torch.Generator().manual_seed(seed)
    u_coeff = torch.randn(n, nt, modes, dtype=torch.complex64, generator=g)
    f = torch.randn(n, 12, generator=g)
    model = SpectralSVR(
        FourierBasis(periods=[1.0, 1.0], complex_funcs=True), LSSVR(kernel="rbf")
    )
    model.train(f, u_coeff, u_time_dependent=True)
    metrics = model.test(f, u_coeff)
    assert set(metrics) == {"spectral", "function value"}


# --------------------------------------------------------------------------- #
# error paths (structural, not fuzzed)
# --------------------------------------------------------------------------- #
@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_train_rejects_dtype_mismatch():
    model = SpectralSVR(FourierBasis(periods=[1.0]), LSSVR())
    with pytest.raises(ValueError, match="must match the basis coeff_dtype"):
        model.train(torch.randn(10, 16), torch.randn(10, 8))  # real target


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_test_rejects_dtype_mismatch():
    model, f, u = _trained_model()
    with pytest.raises(ValueError, match="must match the basis coeff_dtype"):
        model.test(f, u.coeff.real)  # real targets


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_forward_rejects_wrong_x_dims():
    model, f, _ = _trained_model()
    with pytest.raises(ValueError, match="num_points"):
        model.forward(f, torch.randn(20, 2))  # 2D x but 1D modes


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_inverse_before_train_raises():
    model = SpectralSVR(FourierBasis(periods=[1.0]), LSSVR())
    with pytest.raises(RuntimeError, match="has not been trained"):
        model.inverse_coeff(torch.randn(4, 8, dtype=torch.complex64), epochs=1)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_inverse_features_none_when_regressor_prefitted():
    # regressor fitted directly (no SpectralSVR.train) -> features stays None
    reg = LSSVR(kernel="linear", C=10.0)
    reg.fit(torch.randn(12, 16), torch.randn(12, 16))
    model = SpectralSVR(FourierBasis(periods=[1.0]), reg)
    with pytest.raises(RuntimeError, match="features is None"):
        model.inverse_coeff(torch.randn(4, 8, dtype=torch.complex64), epochs=1)


@pytest.mark.no_mms
@SETTINGS
@given(n=st.integers(30, 90), modes=st.integers(4, 12), seed=st.integers(0, 10_000))
def test_train_with_complex_input_features(n, modes, seed):
    # complex input features are converted to interleaved real inside train()
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, n, modes, generator=g, u0=0)
    model = SpectralSVR(FourierBasis(periods=[1.0]), LSSVR(kernel="rbf", C=10.0))
    model.train(ut.coeff, u.coeff)  # complex f
    assert model.features == 2 * modes
