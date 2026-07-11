import torch
from ..basis import Basis, BasisSubType
from ..utils import (
    Number,
    SolverSignatureType,
    implicit_adams_solver,
    resolve_device,
)
from . import Problem
from typing import Literal, Type

ParamInput = Literal["random"] | Number | torch.Tensor


# https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/burgersProject.pdf
# https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/heat_rk4.m
class Burgers(Problem):
    """
    Burger's equation problem for one dimension

    This class defines and generates the Burgers equation problem in one dimension. The functions themselves may be multidimensional but the derivative is only in the 0th mode dimension
    """

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        u0: ParamInput | BasisSubType = "random",
        f: ParamInput | BasisSubType = 0,
        nu: float = 0.01,
        space_domain: slice | None = None,
        time_domain: slice | None = None,
        solver: SolverSignatureType = implicit_adams_solver,
        time_dependent_coeff: bool = True,
        **kwargs,
    ) -> tuple[BasisSubType, BasisSubType]:
        if space_domain is None:
            space_domain = slice(0, 1, 200)
        if time_domain is None:
            time_domain = slice(0, 1, 200)
        if isinstance(modes, int):
            modes = (modes, modes)
        assert n > 0, "number of samples n must be more than 0"
        for dim, m in enumerate(modes):
            assert m > 0, f"number of modes m must be more than 0 at dim {dim}"

        device = resolve_device()

        L = space_domain.stop - space_domain.start
        T: float = time_domain.stop - time_domain.start
        nt = int(time_domain.step)
        dt = T / (nt - 1)
        t = basis.grid(time_domain).flatten().to(device=device)
        assert t[1].sub(t[0]).isclose(torch.tensor(dt)), (
            f"Make sure that the result of generating t is consistent with dt ({dt}) and t[1]-t[0] ({t[1] - t[0]})"
        )
        periods = (T, L)
        if u0 == "random" and f == "random":
            # use method of manufactured solution
            # generate solution itself since u0 just follows from the generate solution
            # time_mode = modes[0]
            spatial_modes = modes[1:]
            u = basis.generate(n, modes, periods=periods, generator=generator)
            res_modes = tuple(slice(0, L, mode) for mode in spatial_modes)

            fst = self.spectral_residual(u, basis(basis.generate_empty(n, modes)), nu)

            u_gen = u
            f_gen = fst
            # convert to timed dependent coeffs
            if time_dependent_coeff:
                u_val = u.get_values(res=(time_domain, *res_modes))
                u_coeff = basis.transform(u_val.flatten(0, 1)).reshape(
                    (n, nt, *spatial_modes)
                )
                u_gen = basis(coeff=u_coeff, time_dependent=True, periods=periods)

                f_val = fst.get_values(res=(time_domain, *res_modes))
                f_coeff = basis.transform(f_val.flatten(0, 1)).reshape(
                    (n, nt, *spatial_modes)
                )
                f_gen = basis(coeff=f_coeff, time_dependent=True, periods=periods)

        else:
            # Only the manufactured-solution path (u0="random", f="random") is
            # implemented. A numerical IVP solver (e.g. ETDRK4) is future work.
            raise NotImplementedError(
                "numerical solver not implemented; use u0='random', f='random'"
            )

        results = (u_gen, f_gen)
        return results

    # rhs spectral formulation adapted from
    # https://math.stackexchange.com/q/3834917 (Gokul, 2020-09-21)
    @staticmethod
    def rhs(
        basis: type[Basis],
        nu: float,
        u_hat: torch.Tensor,
        f_hat: torch.Tensor,
    ) -> torch.Tensor:
        u = basis(u_hat)
        dealias_modes = tuple(int(mode * 1.5) for mode in u.modes)
        u_dealiased = u.resize_modes(dealias_modes, rescale=False)
        u_val = basis.inv_transform(u_dealiased.coeff)
        uu_x_hat_dealiased = 0.5 * basis.transform(u_val**2)
        uu_x = basis(uu_x_hat_dealiased).resize_modes(u.modes, rescale=False).grad()

        u_u_x_hat = uu_x.coeff
        u_xx_hat = u.grad().grad().coeff
        u_t_hat = nu * u_xx_hat + f_hat - u_u_x_hat
        return u_t_hat

    def spectral_residual(
        self, u: BasisSubType, f: BasisSubType, nu: float
    ) -> BasisSubType:
        u_t = u.grad(dim=0, ord=1)

        dealias_modes = tuple(int(mode * 1.5) for mode in u.modes)
        u_dealiased = u.resize_modes(dealias_modes, rescale=False)
        u_val = u.inv_transform(u_dealiased.coeff)
        uu_dealiased = u.copy()
        uu_dealiased.coeff = u.transform(u_val.pow(2).mul(0.5))
        uu_x = uu_dealiased.resize_modes(u.modes, rescale=False).grad(dim=1)

        u_xx = u.grad(dim=1, ord=2)
        nu_u_xx = u_xx
        nu_u_xx.coeff = nu_u_xx.coeff * nu

        residual = u_t + uu_x - nu_u_xx - f
        return residual

    def residual(self, u: BasisSubType, f: BasisSubType, nu: float, res:int = 1000) -> BasisSubType:
        u_val, grid = u.get_values_and_grid(res=res)
        f_val = f.get_values(res=res)
        dt = grid[1, 0, 0] - grid[0, 0, 0]
        dx = grid[0, 1, 1] - grid[0, 0, 1]

        u_t = torch.gradient(u_val, spacing=dt.item(), dim=1, edge_order=2)[0]

        u_x = torch.gradient(u_val, spacing=dx.item(), dim=2, edge_order=2)[0]
        u_xx = torch.gradient(u_x, spacing=dx.item(), dim=2, edge_order=2)[0]

        uu_x = torch.gradient(
            u_val.pow(2).mul(0.5), spacing=dx.item(), dim=2, edge_order=2
        )[0]

        residual_val = u_t + uu_x - nu * u_xx - f_val
        residual = u.copy()
        if u.time_dependent:
            residual.coeff = u.transform(residual_val.flatten(0, 1)).unflatten(
                0, residual_val.shape[0:2]
            )
        else:
            residual.coeff = u.transform(residual_val)

        return residual
