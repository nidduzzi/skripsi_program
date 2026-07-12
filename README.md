# SpectralSVR

Operator learning with spectral bases: model the coefficients of a spectral
series (e.g. Fourier) with least-squares SVR / neural regressors, plus tooling
for generating PDE problem data (antiderivative, viscous Burgers).

## Installation

The project uses [uv](https://docs.astral.sh/uv/). To use it as a dependency:

```bash
uv add git+https://github.com/nidduzzi/SpectralSVR.git
# or, with pip:
pip install "git+https://github.com/nidduzzi/SpectralSVR.git"
```

For development (creates + installs the pinned environment):

```bash
uv sync --extra test --extra lint --extra notebook
```

## Usage

```python
from SpectralSVR import (
    SpectralSVR,
    FourierBasis,
    StandardScaler,
    Antiderivative,
    Burgers,
    LSSVR,
    FNN,
    to_real_coeff,
    to_complex_coeff,
    get_metrics,
)
```

Generate data with a problem, then train a `SpectralSVR` over the spectral
coefficients. Examples live under `notebooks/cases/` — run them block by block
as some tail sections are unfinished.

Burgers data can be produced two ways:

- **Numerical simulation** via a stiff-stable ETDRK4 spectral solver
  (`Burgers().generate(..., u0=<ic>, f=0)`).
- **Method of Manufactured Solutions** (`Burgers().mms_solution(...)`), which
  returns an exact solution/forcing pair for verification.

## Testing

```bash
uv run --extra test pytest
```

Every test is either fuzzed with [Hypothesis](https://hypothesis.readthedocs.io/)
or explicitly marked `@pytest.mark.no_fuzz`, and declares `@pytest.mark.mms` /
`@pytest.mark.no_mms` (enforced by `tests/conftest.py`).

Linting / type-checking:

```bash
uv run --extra lint ruff check SpectralSVR tests
uv run --extra lint mypy SpectralSVR
uv run --extra lint ty check SpectralSVR
```

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
