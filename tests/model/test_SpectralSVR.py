from SpectralSVR import (
    SpectralSVR,
    FourierBasis,
    to_real_coeff,
    to_complex_coeff,
    StandardScaler,
    Antiderivative,
    LSSVR,
)
import pytest
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split
from torchmetrics.functional import symmetric_mean_absolute_percentage_error

# End-to-end scenario: manufactured antiderivative operator (MMS), single
# fixed configuration rather than a fuzzed one.
pytestmark = [pytest.mark.mms, pytest.mark.no_fuzz]


def test_SpectralSVR():
    generator = torch.Generator().manual_seed(42)
    # Generate functions/data
    # spectral density function

    # Compute coefficients for both
    n_coeffs = 2000
    modes = 7
    problem = Antiderivative()
    u_fourier, f_fourier = problem.generate(
        FourierBasis, n_coeffs, modes, generator=generator, u0=0
    )

    # Interpolate f & u
    t = FourierBasis.grid(slice(0, 1, 200))
    f = f_fourier(t)
    f = f.real

    s = FourierBasis.grid(slice(-1, 1, 400))
    u = u_fourier(s)
    u = u.real

    # Add noise

    # Train-test split
    df = TensorDataset(to_real_coeff(f_fourier.coeff), u, u_fourier.coeff)
    df_train, df_test = random_split(df, (0.8, 0.2), generator=generator)

    # Scale inputs based on training inputs
    f_train, u_train, u_coeff_train = df_train[:]
    scaler = StandardScaler().fit(f_train)
    f_train = scaler.transform(f_train)
    f_test, u_test, u_coeff_test = df_test[:]
    f_test = scaler.transform(f_test)

    # Train svm
    periods = [1.0]
    model = SpectralSVR(basis=FourierBasis(periods=periods), regressor=LSSVR())
    print(f"f_train.shape:, {f_train.shape}")
    print(f"u_coeff_train.shape: {u_coeff_train.shape}")
    model.train(f_train, u_coeff_train)

    # Test
    u_coeff_pred = model.regressor.predict(f_test)
    u_coeff_pred = to_complex_coeff(u_coeff_pred)
    smape = symmetric_mean_absolute_percentage_error(
        to_real_coeff(u_coeff_pred), to_real_coeff(u_coeff_test)
    )

    assert torch.isclose(torch.tensor(0.0), smape, atol=96e-3), (
        f"coefficient evaluation smape too high ({smape})"
    )

    u_pred = model.forward(f_test, s).real
    # calculate smape
    smape = symmetric_mean_absolute_percentage_error(u_pred, u_test)
    assert torch.isclose(torch.tensor(0.0), smape, atol=11e-2), (
        f"prediction evaluation smape too high ({smape})"
    )
