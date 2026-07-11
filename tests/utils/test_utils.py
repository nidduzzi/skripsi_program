from SpectralSVR.utils import (
    DEFAULT_METRICS,
    get_metrics,
    resolve_device,
    to_complex_coeff,
    to_real_coeff,
)
import torch


def test_real_invertible_even():
    # even number of coefficients
    a = torch.rand((10, 20))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    invertible = torch.allclose(a, a_real)
    assert invertible, (
        f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"
    )


def test_real_invertible_odd():
    # Odd number of coefficients
    a = torch.rand((10, 18))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    # get rid of the padding
    mask = a_real != 0.0
    mask = mask.sum(dim=0) != 0
    a_real = a_real[:, mask]
    invertible = torch.allclose(a, a_real)
    assert invertible, (
        f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"
    )


def test_complex_invertible_even():
    # even number of coefficients
    c = torch.rand((10, 10), dtype=torch.complex64)
    c_real = to_real_coeff(c)
    c_complex = to_complex_coeff(c_real)
    invertible = torch.equal(c, c_complex)
    assert invertible, (
        f"c with shape {c.shape} and c_complex with shape {c_complex.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, c_real has shape {c_real.shape}"
    )


def test_complex_invertible_odd():
    # Odd number of coefficients
    c = torch.rand((10, 9), dtype=torch.complex64)
    c_real = to_real_coeff(c)
    c_complex = to_complex_coeff(c_real)
    invertible = torch.equal(c, c_complex)
    assert invertible, (
        f"c with shape {c.shape} and c_complex with shape {c_complex.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, c_real has shape {c_real.shape}"
    )
    print("Test Passed")


def test_resolve_device_explicit():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_default_matches_availability():
    resolved = resolve_device(None)
    expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert resolved == expected


def test_get_metrics_default_keys():
    preds = torch.randn(10, 3)
    targets = torch.randn(10, 3)
    m = get_metrics(preds, targets)
    assert set(m) == set(DEFAULT_METRICS) | {"pred_nan_sum"}
    assert m["mse"] >= 0.0


def test_get_metrics_single_sample_r2_is_nan():
    m = get_metrics(torch.randn(1, 3), torch.randn(1, 3))
    assert m["r2"] != m["r2"]  # NaN


def test_get_metrics_injected_metric_set():
    preds = torch.zeros(4, 2)
    targets = torch.zeros(4, 2)
    m = get_metrics(preds, targets, metrics={"custom": lambda p, t: (p - t).abs().sum()})
    assert m["custom"] == 0.0
    assert "mse" not in m
    assert "pred_nan_sum" in m


# TODO: test scale_to_standard
# TODO: test reduce_coeff
