import pytest
from skripsi_program import FourierBasis
from skripsi_program import to_complex_coeff, to_real_coeff
import torch


def test_basis():
    # Generate Signal
    ## sampling rate
    sr = 100
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(0, 1, ts)

    # function 1
    freq = 1.0
    f1 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f1 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f1 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    # function 2
    freq = 1.0
    f2 = 2 * torch.sin(2 * torch.pi * freq * t)
    freq = 5
    f2 += torch.sin(2 * torch.pi * freq * t)
    freq = 10
    f2 += 0.3 * torch.sin(2 * torch.pi * freq * t)

    f = f1.unsqueeze(0)
    # f = torch.stack((f1, f2))
    f = f * (1 + 0j)  # cast to complex

    # Get coefficients and create basis
    coeff = FourierBasis.transform(f)
    basis = FourierBasis(coeff)
    assert basis.modes is not None, "Basis modes is None, it shouldn't be since coeff was passed into FourierBasis"
    # derivative
    k = FourierBasis.waveNumber(basis.modes[0])
    f_coeff = coeff * 2j * torch.pi * k.T
    f_basis = FourierBasis(f_coeff)

    # Odd samples
    # Generate Signal
    ## sampling rate
    sr = 9
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(0, 1, ts)
    # function 1
    freq = 1.0
    f1 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f1 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f1 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    f1 = f1.unsqueeze(0) * (1 + 0j)

    coeff = FourierBasis.transform(f1)
    assert (
        coeff.ndim == 2
    ), "coeff needs to have two dimensions, the first the sample and the second the modes for the coefficients"
    coeff_real = to_real_coeff(coeff)
    coeff_complex = to_complex_coeff(coeff_real)
    invertible = torch.equal(coeff_complex, coeff)
    assert invertible, f"coeff_complex with shape {coeff_complex.shape} and coeff with shape {coeff.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, coeff_real has shape {coeff_real.shape}"
    # Interpolate and and compare f2
    ## sampling rate
    sr = 150
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(-1, 1, ts)
    freq = 1.0
    f3 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f3 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f3 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    f3 = f3 * (1 + 0j)  # cast to complex


    f3_pred = basis(t)[0]
    assert (
        f3_pred.shape == f3.shape
    ), f"f3_pred has shape {f3_pred.shape} and f3 has shape {f3.shape}, both need to have the same shape"

    # Compare prediction with real function
    mse = torch.norm((f3_pred - f3), 2)
    assert torch.isclose(
        torch.tensor(0.0), mse, atol=1e-3
    ), f"interpolation mse too high ({mse})"
