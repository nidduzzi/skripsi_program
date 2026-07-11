import torch
from ..basis import Basis, BasisSubType
from ..utils import (
    Number,
    SolverSignatureType,
    etdrk4_solver,
    implicit_adams_solver,
    resolve_device,
)
from . import Problem
from typing import Literal, Type

ParamInput = Literal["random"] | Number | torch.Tensor


# References:
# - Viscous Burgers background:
#   https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/burgersProject.pdf
#   https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/heat_rk4.m
# - Numerical solver: ETDRK4 (see SpectralSVR.utils.etdrk4_solver for full
#   references: Kassam & Trefethen 2005; Cox & Matthews 2002; the etdrk4cp,
#   rkstiff and SciML Burgers-spectral benchmarks).
# - Exact solutions for testing use the Cole-Hopf transformation: theta solves
#   the heat equation and u = -2*nu*theta_x/theta solves unforced Burgers.
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
        if isinstance(u0, str) and u0 == "random" and isinstance(f, str) and f == "random":
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
            # Numerical initial-value problem: integrate viscous Burgers with a
            # stiff-stable ETDRK4 spectral solver to produce simulated data.
            if not time_dependent_coeff:
                raise NotImplementedError(
                    "numerical Burgers only supports time_dependent_coeff=True"
                )
            u_gen, f_gen = self._solve_spectral(
                basis, n, modes, nu, u0, f, periods, t, nt, device, generator
            )

        results = (u_gen, f_gen)
        return results

    @staticmethod
    def _solve_spectral(
        basis: Type[BasisSubType],
        n: int,
        modes: tuple[int, ...],
        nu: float,
        u0: "ParamInput | BasisSubType",
        f: "ParamInput | BasisSubType",
        periods: tuple[float, ...],
        t: torch.Tensor,
        nt: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Integrate 1D viscous Burgers with ETDRK4 and package the result.

        Basis-agnostic: the diagonal linear operator ``nu * d2/dx2`` and the
        spectral derivative for the nonlinear term ``0.5 (u^2)_x`` are built from
        the basis's ``derivative_eigenvalues``; the transforms use the generic
        ``transform``/``inv_transform``. Works for any basis whose differentiation
        is diagonal (Fourier, spherical harmonics). Solves
        ``u_t + 0.5 (u^2)_x = nu u_xx + forcing`` on spatial period
        ``L = periods[1]`` with ``ns = modes[1]`` spatial modes.
        """
        if len(modes) != 2:
            raise NotImplementedError("numerical Burgers is implemented for 1D space")
        ns = modes[1]
        length = periods[1]

        d1 = basis.derivative_eigenvalues(ns, length, ord=1)
        d2 = basis.derivative_eigenvalues(ns, length, ord=2)
        if d1 is None or d2 is None:
            raise NotImplementedError(
                f"{basis.__name__} differentiation is not diagonal; the ETDRK4 "
                "solver requires a diagonally-differentiable basis (e.g. Fourier)"
            )
        d1 = d1.to(device=device)
        linear = (nu * d2).to(device=device)  # nu * u_xx eigenvalues
        mask = basis.dealias_mask(ns)
        dealias = mask.to(device=device) if mask is not None else 1.0

        # initial condition: random smooth field, a constant, or explicit values
        if isinstance(u0, str) and u0 == "random":
            v0 = basis.generate_coeff(n, ns, generator=generator).to(device=device)
        elif isinstance(u0, Number):
            field = torch.full((n, ns), float(u0), device=device) + 0j
            v0 = basis.transform(field)
        elif isinstance(u0, torch.Tensor):
            field = u0.to(device=device)
            if field.ndim == 1:
                field = field.unsqueeze(0).expand(n, ns)
            assert field.shape == (n, ns), (
                f"u0 values must be ({n}, {ns}) or ({ns},), got {tuple(field.shape)}"
            )
            v0 = basis.transform((field + 0j).clone())
        else:
            raise NotImplementedError(
                "numerical Burgers supports u0='random', a constant, or a value tensor"
            )

        # constant forcing spectrum (0 by default)
        if isinstance(f, Number):
            f_field = torch.full((n, ns), float(f), device=device) + 0j
            f_hat = basis.transform(f_field)
        else:
            raise NotImplementedError("numerical Burgers supports a constant forcing f")

        def nonlinear(ti: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # -0.5 (u^2)_x  in spectral space, with optional de-aliasing
            u = basis.inv_transform(v).real.to(v.dtype)
            return -0.5 * d1 * basis.transform(u * u) * dealias + f_hat

        sol = etdrk4_solver(linear, nonlinear, v0, t.to(device=device))  # (nt, n, ns)
        u_coeff = sol.movedim(0, 1).to(basis.coeff_dtype)  # (n, nt, ns)
        u_gen = basis(coeff=u_coeff, time_dependent=True, periods=periods)

        f_coeff = f_hat.unsqueeze(1).expand(n, nt, ns).to(basis.coeff_dtype).clone()
        f_gen = basis(coeff=f_coeff, time_dependent=True, periods=periods)
        return u_gen, f_gen

    def mms_solution(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: tuple[int, ...],
        nu: float,
        periods: tuple[float, ...],
        generator: torch.Generator | None = None,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Method of Manufactured Solutions for 1D viscous Burgers.

        Pick an arbitrary band-limited solution ``u`` (a random Fourier field)
        and let the forcing be ``f = L(u)`` where ``L`` is the Burgers operator
        :meth:`spectral_residual` (with zero forcing). Then ``u`` exactly solves
        ``u_t + u u_x - nu u_xx = f`` by construction, so ``spectral_residual``
        of the returned pair is zero to machine precision.

        ``u`` is band-limited to half the mode budget so the nonlinear term
        (which doubles the highest frequency) stays resolved and alias-free.
        Returned as non-time-dependent 2D (time, space) Fourier fields.
        """
        if len(modes) != 2:
            raise NotImplementedError("MMS Burgers is implemented for 1D space")
        half = tuple(max(1, m // 2) for m in modes)
        u = basis.generate(n, half, periods=periods, generator=generator).resize_modes(
            modes, rescale=False
        )
        u = basis(coeff=u.coeff, periods=periods)
        zero_forcing = basis(basis.generate_empty(n, modes), periods=periods)
        f = self.spectral_residual(u, zero_forcing, nu)
        return u, f

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
        """Spectral residual of ``u_t + u u_x - nu u_xx - f``.

        ``u`` and ``f`` are a non-time-dependent Fourier field whose first mode
        axis is time and second is space. Everything is expressed through the
        generic :class:`Basis` interface (``grad`` for exact spectral
        derivatives, and ``transform``/``inv_transform`` which round-trip), so
        this works for any spectral basis, not just Fourier. For a band-limited
        manufactured solution the residual is zero to machine precision.
        """
        u_t = u.grad(dim=0, ord=1)
        u_x = u.grad(dim=1, ord=1)
        u_xx = u.grad(dim=1, ord=2)

        # nonlinear term u * u_x formed in physical space via the basis
        # transforms (transform . inv_transform is the identity, so no manual
        # rescaling is needed).
        u_val = u.inv_transform(u.coeff)
        u_x_val = u.inv_transform(u_x.coeff)
        uu_x = u.copy()
        uu_x.coeff = u.transform(u_val * u_x_val)

        nu_u_xx = u_xx.copy()
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
