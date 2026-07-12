"""Test-only glue: Hypothesis settings + the library's exact solution families.

The exact/manufactured solution builders now live in the library
(:mod:`SpectralSVR.problems.exact`) so they can seed benchmarks and demos too;
they are re-exported here for the existing ``from _exact import ...`` test
imports, alongside the test-only Hypothesis ``SETTINGS``.
"""

from hypothesis import HealthCheck, settings

from SpectralSVR.problems.exact import (
    burgers_periodic_manufactured as burgers_periodic_manufactured,
    cole_hopf_params as cole_hopf_params,
    cole_hopf_u as cole_hopf_u,
    heat_mode_params as heat_mode_params,
    heat_mode_u_and_ux as heat_mode_u_and_ux,
    space_grid as space_grid,
)

SETTINGS = settings(
    deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
)
