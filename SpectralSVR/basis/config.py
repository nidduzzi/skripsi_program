"""SpectralConfig: the coefficient-independent configuration of a basis.

A basis wears several hats -- a template/spec, a coefficient container, and a
plotter. This value object carves out the *spec* hat: the geometry and
evaluation knobs that do not depend on any particular set of coefficients
(the domain each axis lives on, how that domain is sampled, and the
memory/approximation tuning for transforms).

Keeping it separate means a basis used purely as a template (e.g. a model's
untrained basis, which holds no coefficients) still has a well-defined domain:
domain resolution takes an explicit mode count instead of reading it off the
coefficient tensor's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import (
    DomainAxis,
    DomainInputType,
    domain_lengths,
    domainInputType_to_tuple,
)
from .sampling import SamplingInputType, SamplingScheme, samplings_to_tuple
from .strategy import DEFAULT_EVALUATION_STRATEGY, EvaluationStrategy


@dataclass(frozen=True, kw_only=True)
class SpectralConfig:
    """Coefficient-independent basis configuration (domain, sampling, strategy).

    ``domain`` and ``sampling`` are stored as given -- a single value broadcast to
    every axis, or one per axis -- and resolved to a concrete per-axis tuple only
    on demand, against an explicit mode count. (A domain may be periodic on one
    axis and not another, so node placement is per-axis too.)
    """

    sampling: SamplingInputType
    domain: DomainInputType = None
    strategy: EvaluationStrategy = DEFAULT_EVALUATION_STRATEGY

    def resolve_domain(self, modes: tuple[int, ...]) -> tuple[DomainAxis, ...]:
        """Broadcast the raw domain to one ``(start, stop)`` pair per axis."""
        return domainInputType_to_tuple(self.domain, modes)

    def resolve_sampling(
        self, ndim: int, default: SamplingScheme
    ) -> tuple[SamplingScheme, ...]:
        """Broadcast the sampling input to one scheme per axis."""
        return samplings_to_tuple(self.sampling, ndim, default)

    def lengths(self, modes: tuple[int, ...]) -> tuple[float, ...]:
        """Per-axis length ``stop - start`` for the given mode count."""
        return domain_lengths(self.resolve_domain(modes))

    def with_domain(self, domain: DomainInputType) -> "SpectralConfig":
        """A copy with a different domain (this object is immutable)."""
        return replace(self, domain=domain)
