"""Tests for SpectralConfig and the domain helpers it wraps."""

import pytest

from SpectralSVR import FourierBasis
from SpectralSVR.basis import SpectralConfig
from SpectralSVR.basis.domain import domainInputType_to_tuple
from SpectralSVR.basis.sampling import PeriodicUniform
from SpectralSVR.basis.strategy import DEFAULT_EVALUATION_STRATEGY


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_config_resolve_domain_broadcasts_single_pair():
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=(-1.0, 1.0))
    assert cfg.resolve_domain((8, 4)) == ((-1.0, 1.0), (-1.0, 1.0))


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_config_lengths():
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=((0.0, 2.0), (1.0, 4.0)))
    assert cfg.lengths((8, 4)) == (2.0, 3.0)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_config_with_domain_is_immutable_copy():
    cfg = SpectralConfig(sampling=PeriodicUniform(), domain=(0.0, 1.0))
    other = cfg.with_domain((-1.0, 1.0))
    assert cfg.domain == (0.0, 1.0)  # original untouched
    assert other.domain == (-1.0, 1.0)
    assert other.sampling is cfg.sampling  # other fields carried over
    assert other.strategy is DEFAULT_EVALUATION_STRATEGY


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_domain_setter_rebuilds_config():
    basis = FourierBasis(FourierBasis.generate_empty(1, (8, 4)), domain=(0.0, 1.0))
    basis.domain = ((0.0, 2.0), (0.0, 3.0))
    assert basis.domain == ((0.0, 2.0), (0.0, 3.0))
    assert basis.lengths == (2.0, 3.0)


@pytest.mark.no_fuzz
@pytest.mark.no_mms
def test_domain_malformed_pair_raises():
    with pytest.raises(ValueError, match="unpack"):
        domainInputType_to_tuple((1.0, 2.0, 3.0), (8,))  # ty: ignore[invalid-argument-type]


@pytest.mark.no_fuzz
@pytest.mark.no_mms
@pytest.mark.parametrize("bad", [(1.0, 1.0), (2.0, 1.0)])
def test_domain_degenerate_or_inverted_raises(bad):
    # zero-length or inverted intervals would divide by zero / flip frequency sign
    with pytest.raises(ValueError, match="positively-oriented interval"):
        domainInputType_to_tuple(bad, (8,))
