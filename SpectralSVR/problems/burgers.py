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
        # x = (
        #     basis.grid(slice(space_domain.start, space_domain.stop, modes[0]))
        #     .flatten(0, -2)
        #     .to(device=device)
        # )
        T: float = time_domain.stop - time_domain.start
        nt = int(time_domain.step)
        # nt = int(T / (0.01 * nu) + 2)
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
            # use ivp solution
            raise NotImplementedError("numerical solver not implemented at this point")
            # TODO: use Exponential Time Differencing RK4 (ETDRK4) solver for more stable and accurate results
            # https://matematicas.uclm.es/cedya09/archive/textos/129_de-la-Hoz-Mendez-F.pdf
            #
            # ignore time modes
            # spatial_modes = modes[1:]
            # u_gen, f_gen = self.solve_numerically(
            #     basis,
            #     n,
            #     modes,
            #     nu,
            #     u0,
            #     f,
            #     periods,
            #     x,
            #     t,
            #     solver,
            #     nt,
            #     time_dependent_coeff,
            #     device,
            #     generator,
            #     **kwargs,
            # )

        results = (u_gen, f_gen)
        return results

    # @staticmethod
    # def _get_initial_value(
    #     basis: Type[BasisSubType],
    #     n: int,
    #     modes: tuple[int, ...],
    #     u0: ParamInput | BasisSubType,
    #     periods: tuple[float, ...],
    #     generator,
    #     **kwargs,
    # ) -> BasisSubType:
    #     out: BasisSubType
    #     if isinstance(u0, str):
    #         match u0:
    #             case "random":
    #                 # random first space coefficients
    #                 out = basis.generate(
    #                     n, modes, generator=generator, periods=periods, **kwargs
    #                 )
    #             case _:
    #                 raise RuntimeError(f"parameter value of u0 ({u0}) is invalid")
    #     elif isinstance(u0, Number):
    #         u0_const = float(u0)
    #         out = basis(basis.generate_empty(n, modes), periods=periods, **kwargs)
    #         if out.coeff.is_complex():
    #             out.coeff[:, ..., 0] = torch.tensor(u0_const + 0j)
    #         else:
    #             out.coeff[:, ..., 0] = u0_const
    #     elif isinstance(u0, torch.Tensor):
    #         out = basis(u0).resize_modes(modes, rescale=False)
    #     elif isinstance(u0, Basis):
    #         out = u0
    #     else:
    #         raise RuntimeError("Invalid u0 value")
    #     return out
    #
    # @classmethod
    # def solve_numerically(
    #     cls,
    #     basis: Type[BasisSubType],
    #     n: int,
    #     modes: tuple[int, ...],
    #     nu: float,
    #     _u0: ParamInput | BasisSubType,
    #     f: ParamInput | BasisSubType,
    #     periods: tuple[float, ...],
    #     x: torch.Tensor,
    #     t: torch.Tensor,
    #     solver: SolverSignatureType,
    #     nt: int,
    #     timedependent_solution: bool,
    #     device,
    #     generator,
    #     **kwargs,
    # ) -> tuple[BasisSubType, BasisSubType]:
    #     # Setup solution using initial condition
    #     u0 = cls._get_initial_value(basis, n, modes, _u0, periods, generator, **kwargs)
    #     # Setup forcing term
    #     if isinstance(f, str):
    #         match f:
    #             case "random":
    #                 fst = basis.generate(
    #                     n,
    #                     (*modes, *modes),
    #                     generator=generator,
    #                     periods=periods,
    #                     **kwargs,
    #                 )
    #             case _:
    #                 raise RuntimeError(f"parameter value of f ({f}) is invalid")
    #     elif isinstance(f, Number):
    #         f_const = float(f)
    #         fst = basis(basis.generate_empty(n, (1, *modes)), periods=periods, **kwargs)
    #         if fst.coeff.is_complex():
    #             fst.coeff[:, ..., 0] = torch.tensor(f_const + 0j)
    #         else:
    #             fst.coeff[:, ..., 0] = f_const
    #     elif isinstance(f, torch.Tensor):
    #         fst = basis(f).resize_modes((*modes, *modes), rescale=False)
    #     elif isinstance(f, Basis):
    #         fst = f
    #     else:
    #         raise RuntimeError("Invalid f value")
    #
    #     u0.coeff = u0.coeff.to(device=device)
    #     fst.coeff = fst.coeff.to(device=device)
    #     print(f"generating with {len(t)} time steps")
    #
    #     def f_func(t: torch.Tensor, x: torch.Tensor = x):
    #         x = x.tile((1, 2))
    #         x[:, 0] = t
    #         return basis.transform(fst(x).reshape((-1, modes[0])))
    #
    #     def rhs_func(t: torch.Tensor, y0: torch.Tensor):
    #         y0 = to_complex_coeff(y0)
    #         return to_real_coeff(cls.rhs(basis, nu, y0, f_func(t)))
    #
    #     u_hat = solver(rhs_func, to_real_coeff(u0.coeff), t)
    #     if basis.coeff_dtype.is_complex:
    #         u_shape = u_hat.shape
    #         u_hat = to_complex_coeff(u_hat.flatten(0, 1)).reshape(
    #             (*u_shape[:2], *u0.coeff.shape[1:])
    #         )
    #
    #     u_hat = u_hat.movedim(0, 1)
    #     if timedependent_solution:
    #         u = basis(u_hat, **kwargs, time_dependent=True)
    #     else:
    #         u_hat = u_hat.reshape((n * nt, modes[0]))
    #         u = basis(
    #             basis.transform(basis.inv_transform(u_hat).reshape((n, nt, modes[0]))),
    #             **kwargs,
    #         )  # .resize_modes((*modes,) * 2)
    #
    #     u.coeff = u.coeff.cpu()
    #     fst.coeff = fst.coeff.cpu()
    #     return (u, fst)
    #
    # addapted from
    # @MISC {3834917,
    #     TITLE = {Solving Viscous Burgers using spectral method},
    #     AUTHOR = {Gokul (https://math.stackexchange.com/users/827487/gokul)},
    #     HOWPUBLISHED = {Mathematics Stack Exchange},
    #     NOTE = {URL:https://math.stackexchange.com/q/3834917 (version: 2020-09-21)},
    #     EPRINT = {https://math.stackexchange.com/q/3834917},
    #     URL = {https://math.stackexchange.com/q/3834917}
    # }

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
