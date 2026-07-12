"""Tests for the NUFFT-backed non-uniform evaluation path.

The NUFFT is validated against the exact dense (matmul) evaluation, which is
the ground truth here (not a manufactured PDE solution), so these are no_mms.
Kaiser-Bessel gridding is approximate, hence the loose tolerance.
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from SpectralSVR import FourierBasis
from SpectralSVR.basis._nufft import nufft_available, nufft_evaluate

CPU = torch.device("cpu")
SETTINGS = settings(
    deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow]
)


def _exact_evaluate(coeff, x, periods):
    # exact dense path: disable the NUFFT for this call only
    return FourierBasis.evaluate(
        coeff, x, periods=periods, nufft_threshold=float("inf")
    )


@pytest.mark.no_mms
@SETTINGS
@given(
    modes=st.integers(8, 48),
    rows=st.integers(1, 3),
    npts=st.integers(20, 200),
    length=st.floats(0.5, 3.0),
    seed=st.integers(0, 10_000),
)
def test_nufft_matches_dense_1d(modes, rows, npts, length, seed):
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(rows, modes, generator=g).to(torch.complex128)
    x = (torch.rand(npts, 1, generator=g, dtype=torch.float64) * length)
    approx = nufft_evaluate(coeff, x, (length,))
    exact = _exact_evaluate(coeff, x, (length,))
    assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()


@pytest.mark.no_mms
@SETTINGS
@given(
    m=st.integers(8, 24),
    npts=st.integers(20, 120),
    seed=st.integers(0, 10_000),
)
def test_nufft_matches_dense_2d(m, npts, seed):
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(1, (m, m), generator=g).to(torch.complex128)
    x = torch.rand(npts, 2, generator=g, dtype=torch.float64) * torch.tensor([1.3, 0.7])
    approx = nufft_evaluate(coeff, x, (1.3, 0.7))
    exact = _exact_evaluate(coeff, x, (1.3, 0.7))
    assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_evaluate_switches_to_nufft_above_threshold():
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator().manual_seed(0)
        coeff = FourierBasis.generate_coeff(1, 32, generator=g).to(torch.complex128)
        x = torch.rand(2000, dtype=torch.float64) * 1.3

        # per-instance overrides -- no global state change
        exact = FourierBasis(coeff, periods=1.3, nufft_threshold=float("inf"))(
            x, device=CPU
        )
        approx = FourierBasis(coeff, periods=1.3, nufft_threshold=0)(x, device=CPU)

        assert (approx - exact).abs().max() < 5e-2 * exact.abs().max()
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.no_mms
@SETTINGS
@given(
    modes=st.integers(4, 32),
    rows=st.integers(1, 3),
    npts=st.integers(1, 200),
    seed=st.integers(0, 10_000),
)
def test_dense_evaluate_chunking_is_exact(modes, rows, npts, seed):
    # Chunking the exact path must not change the result vs a single block.
    g = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(rows, modes, generator=g).to(torch.complex128)
    x = torch.rand(npts, generator=g, dtype=torch.float64)

    # exact matmul (inf threshold); one block vs many small chunks -- per instance
    whole = FourierBasis(
        coeff, periods=1.0, nufft_threshold=float("inf"), dense_chunk_elems=10**12
    )(x, device=CPU)
    chunked = FourierBasis(
        coeff, periods=1.0, nufft_threshold=float("inf"), dense_chunk_elems=modes
    )(x, device=CPU)
    assert torch.allclose(whole, chunked, atol=1e-12)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_dense_evaluate_chunk_bounds_memory():
    # The exact (matmul) path, chunked, must also stay well under the full
    # dense (npts x modes) basis matrix.
    from memory_profiler import memory_usage

    torch.set_default_dtype(torch.float64)
    try:
        modes = 128
        npts = 400_000
        dense_bytes = npts * modes * 16
        coeff = FourierBasis.generate_coeff(1, modes).to(torch.complex128).cpu()
        x = (torch.rand(npts, dtype=torch.float64)).cpu()
        # exact path (inf threshold), ~1M-entry chunks -- per instance
        basis = FourierBasis(
            coeff, periods=1.0, nufft_threshold=float("inf"), dense_chunk_elems=2**20
        )
        holder: dict = {}

        def run():
            holder["out"] = basis(x, device=CPU)

        baseline = memory_usage(-1, max_usage=True)
        peak = memory_usage((run, (), {}), max_usage=True, interval=0.02)
        delta_bytes = max(0.0, peak - baseline) * 1024 * 1024
        assert holder["out"].shape == (1, npts)
        assert delta_bytes < 0.5 * dense_bytes
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_nufft_available():
    assert nufft_available() is True


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_nufft_evaluate_shape():
    coeff = FourierBasis.generate_coeff(3, 16).to(torch.complex64)
    x = torch.rand(50, 1)
    out = nufft_evaluate(coeff, x, (1.0,))
    assert out.shape == (3, 50)
    assert out.is_complex()


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_nufft_evaluate_does_not_materialize_dense_matrix():
    # The bottleneck the NUFFT fixes: the dense path builds a (npts x modes)
    # complex basis matrix. Here that would be ~0.8 GB; the NUFFT must stay far
    # below that. Run on CPU and measure the peak RSS increase.
    from memory_profiler import memory_usage

    modes = 128
    npts = 400_000
    dense_bytes = npts * modes * 16  # complex128 basis matrix (~0.8 GiB)

    coeff = FourierBasis.generate_coeff(1, modes).to(torch.complex128).cpu()
    x = (torch.rand(npts, 1, dtype=torch.float64) * 1.0).cpu()

    holder: dict = {}

    def run():
        holder["out"] = nufft_evaluate(coeff, x, (1.0,))

    baseline = memory_usage(-1, max_usage=True)
    peak = memory_usage((run, (), {}), max_usage=True, interval=0.02)
    delta_bytes = max(0.0, (peak - baseline)) * 1024 * 1024  # MiB -> bytes

    assert holder["out"].shape == (1, npts)
    assert not torch.isnan(holder["out"]).any()
    # comfortably under the dense matrix that the matmul path would allocate
    assert delta_bytes < 0.5 * dense_bytes, (
        f"NUFFT peak delta {delta_bytes / 1e6:.0f} MB exceeded half the dense "
        f"matrix {dense_bytes / 1e6:.0f} MB"
    )
