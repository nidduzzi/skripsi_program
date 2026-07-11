import torch
from ..basis import BasisSubType
from . import Problem
from typing import Type


class Antiderivative(Problem):
    """
    Antiderivative problem for one dimension

    This class defines and generates the antiderivative problem in one
    dimension. The functions themselves may be multidimensional but the
    derivative is only in the 0th mode dimension
    """

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        u0: float | int | complex = 0.0,
        **kwargs,
    ) -> tuple[BasisSubType, BasisSubType]:
        if isinstance(modes, int):
            modes = (modes,)
        assert n > 0, "number of samples n must be more than 0"
        for i, m in enumerate(modes):
            assert m > 0, f"number of modes m must be more than 0 at dim {i}"
        assert u0 is not None, "integration constant u0 must not be None"
        # generate solution functions
        u = basis.generate(n, modes, generator=generator, **kwargs)
        assert u.coeff is not None, (
            "generated solution functions u should not have None coeff"
        )
        # compute derivative functions
        ut = u.grad()
        # set the integration coefficient
        assert u.coeff is not None and ut.coeff is not None, (
            "Something went wrong, because generated functions have value of None for coeff"
        )
        if isinstance(u0, complex):
            if u.coeff.is_complex():
                u.coeff[:, 0] = torch.tensor(u0)
            else:
                u.coeff[:, 0] = torch.tensor(u0).real
        elif isinstance(u0, float) or isinstance(u0, int):
            if u.coeff.is_complex():
                u.coeff[:, 0] = torch.tensor(u0 + 0j)
            else:
                u.coeff[:, 0] = torch.tensor(u0)
        else:
            u.coeff[:, 0] = u0

        results = (u, ut)
        return results

    def spectral_residual(self, u: BasisSubType, ut: BasisSubType) -> BasisSubType:
        residual = u.grad() - ut
        if residual.coeff is not None:
            residual.coeff[:, 0].mul_(0)

        return residual

    def residual(self, u: BasisSubType, ut: BasisSubType) -> BasisSubType:
        u_val, grid = u.get_values_and_grid()
        ut_val = ut.get_values()
        dt = grid[1, 0] - grid[0, 0]
        u_grad = torch.gradient(u_val, spacing=dt.item(), dim=1)[0]
        residual_val = u_grad - ut_val
        residual = u.copy()
        residual.coeff = u.transform(residual_val)
        return residual
