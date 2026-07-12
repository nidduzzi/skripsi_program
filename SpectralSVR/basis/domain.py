"""Domain: the per-axis interval a basis lives on.

The old scalar ``period`` is now the axis *length* (stop - start); geometry
(where nodes/evaluation sit) and scale (how coordinates normalise into the
basis's canonical argument) are separated. ``start`` may be negative, positive,
or zero (e.g. Chebyshev's canonical [-1, 1]).
"""

from __future__ import annotations

from typing import Sequence

from typing_extensions import TypeIs

DomainAxis = tuple[float, float]
DomainInputType = DomainAxis | list[DomainAxis] | tuple[DomainAxis, ...] | None


def _is_axis_sequence(
    domain: DomainAxis | Sequence[DomainAxis],
) -> TypeIs[Sequence[DomainAxis]]:
    """Narrow a per-axis sequence of pairs from a single ``(start, stop)`` pair.

    Both are tuples, so they are told apart by their first element: a sequence of
    axes has a pair there; a single axis has a number. An empty input is an
    (empty) sequence of axes -- unambiguously not a single ``(start, stop)`` pair,
    which always has exactly two numbers.
    """
    return len(domain) == 0 or isinstance(domain[0], (tuple, list))


def domainInputType_to_tuple(
    domain: DomainInputType, modes: tuple[int, ...]
) -> tuple[DomainAxis, ...]:
    """Normalise a domain input into one ``(start, stop)`` pair per axis.

    ``None`` defaults to the unit interval ``(0.0, 1.0)`` on every axis; a single
    ``(start, stop)`` pair is broadcast to every axis; a sequence of pairs is
    taken axis-for-axis (an empty sequence yields no axes).
    """
    if domain is None:
        return tuple((0.0, 1.0) for _ in range(len(modes)))
    if _is_axis_sequence(domain):
        resolved = tuple((float(start), float(stop)) for start, stop in domain)
    else:
        # a single (start, stop) pair broadcast to every axis; a malformed pair
        # (wrong element count) fails the unpack with a clear ValueError.
        start, stop = domain
        resolved = tuple((float(start), float(stop)) for _ in range(len(modes)))
    for start, stop in resolved:
        # length ``stop - start`` is a denominator throughout the transforms
        # (``(x-start)/L``, ``2*pi*i*k/L``); a degenerate or inverted interval
        # would divide by zero or flip the frequency sign.
        if stop <= start:
            raise ValueError(
                "domain axis must be a positively-oriented interval with "
                f"stop > start, got ({start}, {stop})"
            )
    return resolved


def domain_lengths(domain: tuple[DomainAxis, ...]) -> tuple[float, ...]:
    """Per-axis length ``stop - start`` (the quantity the old ``period`` named)."""
    return tuple(stop - start for start, stop in domain)
