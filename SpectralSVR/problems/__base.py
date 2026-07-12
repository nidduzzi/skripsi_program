import abc
import torch
from ..basis import BasisSubType


class Problem(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def axis_diff(
        u: torch.Tensor, spacing: float, dim: int, order: int, periodic: bool
    ) -> torch.Tensor:
        """First/second finite difference along one axis.

        Whether the axis wraps is passed in (read from the sampling scheme the
        values were sampled on -- ``scheme.is_periodic``), not assumed. On a
        periodic axis the stencil rolls around, staying second-order accurate at
        the endpoints; on a non-periodic axis it uses ``torch.gradient``
        (one-sided at the ends). get_values now samples on the basis's own
        (half-open, for Fourier) grid, so the periodic branch is exact.
        """
        if order not in (1, 2):
            raise ValueError(f"axis_diff supports order 1 or 2, got {order}")
        if periodic:
            if order == 1:
                return (u.roll(-1, dim) - u.roll(1, dim)) / (2 * spacing)
            return (u.roll(-1, dim) - 2 * u + u.roll(1, dim)) / (spacing * spacing)
        d1 = torch.gradient(u, spacing=spacing, dim=dim, edge_order=2)[0]
        if order == 1:
            return d1
        return torch.gradient(d1, spacing=spacing, dim=dim, edge_order=2)[0]

    @abc.abstractmethod
    def generate(
        self,
        basis: type[BasisSubType],
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[BasisSubType, ...]:
        pass

    @abc.abstractmethod
    def spectral_residual(self, u: BasisSubType, *args, **kwargs) -> BasisSubType:
        pass

    @abc.abstractmethod
    def residual(self, u: BasisSubType, *args, **kwargs) -> BasisSubType:
        pass
