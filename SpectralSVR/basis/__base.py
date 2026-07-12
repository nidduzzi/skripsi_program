import torch
import abc
from typing_extensions import TYPE_CHECKING, Self, Literal, TypeVar, overload
import logging
from ..utils import Number, resize_modes, interpolate_tensor, resolve_device
from .sampling import SamplingScheme

if TYPE_CHECKING:
    from types import ModuleType
    from matplotlib.axes import Axes
# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x

logger = logging.getLogger(__name__)
ResType = int | slice | tuple[slice, ...]
EvaluationModeType = Literal["inverse transform", "basis"]
AutoEvaluationModeType = Literal["auto"] | EvaluationModeType
PeriodsInputType = Number | list[Number] | tuple[Number, ...] | None


def periodsInputType_to_tuple(
    periods: PeriodsInputType, modes: tuple[int, ...]
) -> tuple[float, ...]:
    if isinstance(periods, list) or isinstance(periods, tuple):
        periods = tuple(float(period) for period in periods)
    else:
        if periods is None:
            periods = 1.0
        periods = float(periods)
        periods = tuple(periods for _ in range(len(modes)))
    return periods


def transformResType_to_tuple(
    res: ResType | None, modes: tuple[int, ...], periods: tuple[float, ...]
) -> tuple[slice, ...]:
    assert len(periods) == len(modes), "periods should have the same dimension as modes"
    if res is None:
        _res = tuple(
            slice(0, period, mode) for period, mode in zip(periods, modes, strict=True)
        )
    elif isinstance(res, int):
        _res = tuple(slice(0, period, res) for period in periods)
    elif isinstance(res, slice):
        _res = tuple(res for mode in modes)
    else:
        assert len(res) == len(modes), (
            f"expected res of length {len(modes)} but got length {len(res)}"
        )
        _res = res
    return _res


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    _coeff: torch.Tensor
    coeff_dtype: torch.dtype

    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        periods: PeriodsInputType = 1,
        complex_funcs: bool = False,
        time_dependent: bool = False,
        sampling: "SamplingScheme | None" = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if not hasattr(self, "coeff_dtype"):
            raise NotImplementedError("Subclasses must define 'coeff_dtype'")
        self.coeff = coeff
        self.periods = periods
        self._complex_funcs = complex_funcs
        self.time_dependent = time_dependent
        # injected node-placement strategy; subclass supplies its canonical one
        self.sampling: SamplingScheme = (
            sampling if sampling is not None else self.default_sampling()
        )

    @staticmethod
    @abc.abstractmethod
    def default_sampling() -> "SamplingScheme":
        """The basis's canonical sampling scheme, used when none is injected."""
        ...

    @property
    def coeff(self) -> torch.Tensor:
        return self._coeff  # pragma: no cover  (subclasses shadow this getter)

    @coeff.setter
    @abc.abstractmethod
    def coeff(self, coeff: torch.Tensor | None):
        if coeff is None or coeff.numel() == 0:
            self._coeff = torch.empty(0, dtype=self.coeff_dtype)
        else:
            assert coeff.ndim >= 2, (
                "coeff needs to be at least a two dimensional tensor of coefficients"
            )
            self._coeff = coeff

    @property
    def modes(self) -> tuple[int, ...]:
        if self.ndim < 1:
            return (0,)
        else:
            return self.get_modes(self.coeff, self.time_dependent)

    @staticmethod
    def get_modes(coeff: torch.Tensor, time_dependent: bool) -> tuple[int, ...]:
        if time_dependent:
            return tuple(coeff.shape[2:])
        else:
            return tuple(coeff.shape[1:])

    @property
    def ndim(self) -> int:
        coeff_ndim = self.coeff.ndim
        if self.time_dependent:
            return coeff_ndim - 2
        else:
            return coeff_ndim - 1

    @property
    def time_size(self):
        if not self.time_dependent:
            return 0
        return self.coeff.shape[1]

    @property
    def periods(self) -> tuple[float, ...]:
        return periodsInputType_to_tuple(
            self._periods,
            self.coeff.shape[1:],
        )

    @periods.setter
    def periods(
        self,
        periods: PeriodsInputType,
    ):
        if len(self.coeff.shape) > 0:
            periods = periodsInputType_to_tuple(periods, self.coeff.shape[1:])
        self._periods = periods

    @staticmethod
    @abc.abstractmethod
    def prefered_evaluation_mode() -> EvaluationModeType:
        """
        prefered_evaluation_mode

        returns the prefered evaluation mode of a basis

        Returns:
            EvaluationModeType -- the prefered evaluation mode
        """
        ...

    def _get_values_from_inverse_transform(
        self,
        i: int,
        n: int,
        res: tuple[slice, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n > 0:
            coeff = self.coeff[i : i + n]
        else:
            coeff = self.coeff
        coeff = coeff.to(device=device)

        if self.time_dependent:
            res_spatial = res[1:]
            values = self.inv_transform(
                coeff.flatten(0, 1),
                res=res_spatial,
                periodic=False,
                periods=self.periods[1:],
            ).unflatten(0, coeff.shape[0:2])
            res_t = res[0]
            t = self.grid(res_t).to(device=device)
            index_float = t.flatten() / self.periods[0] * (values.shape[1] - 1)
            values = self.interpolate_time_tensor(values, index_float)
        else:
            res_spatial = res
            values = self.inv_transform(
                coeff,
                res=res_spatial,
                periodic=False,  # TODO: handle periodicity better
                periods=self.periods,
            )

        values = values.to(self.coeff)
        grid = self.grid(res)
        return values, grid

    def _get_values_from_basis_eval(
        self,
        i: int,
        n: int,
        res: tuple[slice, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        res_t = None
        if self.time_dependent:
            assert len(res) > 1, (
                "res list should be more than one element for time dependent coefficients"
            )
            res_t = res[0]
            res = res[1:]

        grid = self.grid(res)
        grid_device = grid.to(device=device)
        if res_t is None:
            grid_t = None
            grid_t_device = None
            grid_shape = tuple(r.step for r in res)
        else:
            grid_t = self.grid(res_t)
            grid_t_device = grid_t.to(device=device)
            grid_shape = tuple(r.step for r in (res_t, *res))

        values = self.__call__(
            grid_device.flatten(0, -2), t=grid_t_device, i=i, n=n
        ).reshape((-1, *grid_shape))
        if res_t is not None:
            # grid with the time coordinates for complete grid
            grid = self.grid((res_t, *res))
        return values, grid

    def _get_res_tuple(
        self,
        res: ResType | None,
    ) -> tuple[slice, ...]:
        evaluation_dim = (self.ndim + 1) if self.time_dependent else self.ndim
        if res is None:
            modes = self.modes
            periods = self.periods
            if self.time_dependent:
                modes = (self.time_size, *modes)
            return tuple(
                slice(0, period, mode)
                for mode, period in zip(modes, periods, strict=False)
            )
        if isinstance(res, int):
            return tuple(slice(0, period, res) for period in self.periods)
        elif isinstance(res, slice):
            return (res,) * evaluation_dim
        return res

    def get_values_and_grid(
        self,
        i=0,
        n=-1,
        res: ResType | None = None,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get_values_and_grid

        computes the values and evaluation grid of the coefficients

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {-1}). The default evaluates all functions
            res {int | slice | tuple[slice,...] | None} -- function discretization resolution and domain (default: {0:period:dimension modes} domain from 0 to the dimensio period on all dimensions with the same number of points as the dimensions modes each). By default if only an int or a single slice is given, every dimension will share the same range and the resolution is based on the number of dimensions.
            evaluation_mode {"auto" | "inverse transform" | "basis"} -- coefficient evaluation mode (default: {"auto"}). Auto will use the inverse transform if the number of evaluations is high or res is not provided
            device {torch.device | None} -- device the evaluations are done on (default: {None}). By default, the function will try to use the GPU and fallback on the CPU.

        Returns:
            tuple[torch.Tensor, torch.Tensor] -- tuple of value and grid respectively of the evaluated functions
        """
        device = resolve_device(device)
        if evaluation_mode == "auto":
            evaluation_mode = self.prefered_evaluation_mode()
        fin_res = self._get_res_tuple(res)

        if evaluation_mode == "inverse transform":
            values, grid = self._get_values_from_inverse_transform(
                i, n, fin_res, device
            )
        else:
            values, grid = self._get_values_from_basis_eval(i, n, fin_res, device)

        return values, grid

    def get_values(
        self,
        i=0,
        n=-1,
        res: ResType | None = None,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        get_values

        computes the values of the coefficients

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {-1}). The default evaluates all functions
            res {int | slice | tuple[slice,...] | None} -- function discretization resolution and domain (default: {0:period:dimension modes} domain from 0 to the dimensio period on all dimensions with the same number of points as the dimensions modes each). By default if only an int or a single slice is given, every dimension will share the same range and the resolution is based on the number of dimensions.
            evaluation_mode {"auto" | "inverse transform" | "basis"} -- coefficient evaluation mode (default: {"auto"}). Auto will use the inverse transform if the number of evaluations is high or res is not provided
            device {torch.device | None} -- device the evaluations are done on (default: {None}). By default, the function will try to use the GPU and fallback on the CPU.

        Returns:
            torch.Tensor -- value of the evaluated functions
        """
        return self.get_values_and_grid(
            i=i, n=n, res=res, evaluation_mode=evaluation_mode, device=device
        )[0]

    def __len__(self) -> int:
        return self.coeff.__len__()

    @staticmethod
    @abc.abstractmethod
    def fn(
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        fn

        evaluate the value of the basis functions

        Arguments:
            x {torch.Tensor} -- the m by ndim matrix of points to evaluate the basis functions at.

        Returns:
            torch.Tensor -- returns a vector using a tensor of the shape {m,modes}
        """
        pass

    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i: int = 0,
        n: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        __call__

        evaluate approximated function at points x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at

        Keyword Arguments:
            t {torch.Tensor | None} -- p time coordinates for time dependent coordinates (default: {None})
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)

        Returns:
            torch.Tensor -- {n, p, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @overload
    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        i=0,
        n=0,
        time_dependent: Literal[True] | bool = True,
        periods: PeriodsInputType = None,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: None = None,
        i=0,
        n=0,
        time_dependent: Literal[False] | bool = False,
        periods: PeriodsInputType = None,
        **kwargs,
    ) -> torch.Tensor: ...

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i: int = 0,
        n: int = 0,
        time_dependent: bool = False,
        periods: PeriodsInputType = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function of coeff at points x

        Arguments:
            coeff {torch.Tensor} -- coefficients of approximated functions
            x {torch.Tensor} -- m points to evaluate approximated function at

        Keyword Arguments:
            t {torch.Tensor | None} -- p time coordinates for time dependent coefficients (default: {None})
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)
            time_dependent {bool} -- whether the coefficients are time dependent or not (default: {False} not time dependent coefficients)

        Returns:
            torch.Tensor -- {n, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def transform(
        f: torch.Tensor,
        res: ResType | None = None,
        periodic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        transform

        compute basis coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of descretized functions to compute the coefficients of.
            res {tuple[slice,...] | None} -- resolution to evaluate the function at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,1).
            periodic {bool} -- whether the evaluation grid should include the ends or not (periodic)

        Returns:
            torch.Tensor -- m vectors of coefficients
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def inv_transform(
        f: torch.Tensor,
        res: ResType | None = None,
        periodic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        inv_transform

        compute function values from dft coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of coefficeints to compute the function values of.
            res {tuple[slice,...] | None} -- resolution to evaluate the coefficients at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,1).
            periodic {bool} -- whether the evaluation grid should include the ends or not (periodic)

        Returns:
            torch.Tensor -- m vectors of function values
        """
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        complex_funcs: bool = False,
        periods: PeriodsInputType = None,
        **kwargs,
    ) -> Self:
        """
        generate

        generate functions using basis functions with random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int,...]} -- number of coefficients in a series.

        Keyword Arguments:
            generator {torch.Generator | None} -- PRNG Generator for reproducability (default: {None})
            random_func {callable} -- random function that generates the coefficients (default: {torch.randn})
            complex_funcs {bool} -- whether the functions generated should be complex or not (default: {False})
            periods {int | float | list[int|float] | tuple[int|float] | None} -- the period for which the coefficients of the basis applies to (default: {None})

        Returns:
            Basis -- n sets of functions with coefficients with the shape (n, modes)
        """

    @classmethod
    @abc.abstractmethod
    def generate_coeff(
        cls, n: int, modes: int | tuple[int, ...], *args, **kwargs
    ) -> torch.Tensor:
        """
        generate_coeff

        generate random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int,...]} -- number of coefficients in each function.

        Returns:
            torch.Tensor -- n sets of coefficients with the shape (n, modes)
        """

    @classmethod
    @abc.abstractmethod
    def generate_empty(
        cls, n: int, modes: int | tuple[int, ...], *args, **kwargs
    ) -> torch.Tensor:
        """
        generate_empty

        generates coefficient array with zero values

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int, ...]} -- number of coefficients in each function

        Raises:
            TypeError: _description_
            TypeError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            torch.Tensor -- n sets of coefficients with the shape (n, modes)
        """

    @abc.abstractmethod
    def grad(self, dim: int = 0, ord: int = 1) -> Self:
        """
        grad

        grad computes the derivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute derivative on (default: {0})
            ord {int} -- order of derivative (default: {1})

        Returns:
            Self -- returns an instance of current basis with derivative coefficients
        """

    @abc.abstractmethod
    def integral(self, dim: int = 0, ord: int = 1) -> Self:
        """
        integral

        integral computes the antiderivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute antiderivative on (default: {0})
            ord {int} -- order of antiderivative (default: {1})

        Returns:
            Self -- returns an instance of current basis with antiderivative coefficients
        """

    @staticmethod
    def derivative_eigenvalues(
        modes: int, period: float, ord: int = 1
    ) -> torch.Tensor | None:
        """Diagonal differentiation multiplier along one axis, or ``None``.

        For a spectral basis whose differentiation operator is *diagonal* in
        coefficient space (Fourier: ``(2*pi*i*k/period)``; spherical harmonics:
        per-degree factors), this returns the length-``modes`` vector that
        ``grad`` multiplies coefficients by (raised to ``ord``). Bases whose
        differentiation is not diagonal (Chebyshev, wavelet) return ``None``.

        This is what lets diagonal-operator solvers (e.g. the ETDRK4 exponential
        integrator) be written basis-agnostically: they build the linear
        operator and spectral derivatives from these eigenvalues via the generic
        ``transform``/``inv_transform`` interface.
        """
        return None

    @staticmethod
    def dealias_mask(modes: int, fraction: float = 2.0 / 3.0) -> torch.Tensor | None:
        """Boolean keep-mask for anti-aliasing a length-``modes`` axis, or ``None``.

        Returns ``None`` for bases without a canonical de-aliasing rule.
        """
        return None

    @abc.abstractmethod
    def copy(self) -> Self:
        """
        copy

        copy returns an instance of the same basis subclass with identical attributes

        Returns:
            Self -- a copied instance of current instance
        """
        return self.__class__(
            coeff=self.coeff,
            periods=self.periods,
            complex_funcs=self._complex_funcs,
            time_dependent=self.time_dependent,
            sampling=self.sampling,
        )

    def __sub__(self, other: Self):
        if isinstance(other, self.__class__):
            result = self.resize_modes(other)
            result.coeff = result.coeff - other.coeff
            return result
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    def __add__(self, other: Self):
        if isinstance(other, self.__class__):
            result = self.resize_modes(other)
            result.coeff = result.coeff + other.coeff
            return result
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    @abc.abstractmethod
    def plot_coefficients(
        self,
        i: int = 0,
        n: int = 1,
        plt: "ModuleType | Axes | None" = None,
        **kwargs,
    ):
        """Plot the basis coefficients (the spectrum) of ``n`` functions from ``i``.

        Each basis renders its own natural view of its coefficients (Fourier:
        magnitude/phase versus wavenumber; Chebyshev/wavelet: their own layouts),
        so this is abstract. Extra keyword arguments are forwarded to matplotlib.
        """

    def plot(
        self,
        i=0,
        n=1,
        res: ResType | None = None,
        plt: "ModuleType | Axes | None" = None,
        complex_scatter=False,
        plot_component: None | Literal["imag", "real"] = None,
        legend: bool = True,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
        **kwargs,
    ):
        """Plot the functions in the basis (delegates to :func:`_plot.plot_basis`).

        Extra keyword arguments are forwarded to the matplotlib call.

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- number of functions after i to plot (default: {1})
            res {int | slice | tuple[slice,...] | None} -- discretization/domain
                (default: 200 points over each period).
            plt -- Axes or pyplot module to draw on (default: pyplot).
            complex_scatter {bool} -- scatter complex values instead (default: False).
            plot_component {None | "imag" | "real"} -- which component (default: both).
            legend {bool} -- add a legend (default: True).
            evaluation_mode -- coefficient evaluation mode (default: "auto").
            device -- device for evaluation (default: auto GPU/CPU).

        Returns:
            The matplotlib artist returned by the underlying plotting call.
        """
        from ._plot import plot_basis

        return plot_basis(
            self,
            i=i,
            n=n,
            res=res,
            plt=plt,
            complex_scatter=complex_scatter,
            plot_component=plot_component,
            legend=legend,
            evaluation_mode=evaluation_mode,
            device=device,
            **kwargs,
        )

    @staticmethod
    def grid(res: ResType = 200) -> torch.Tensor:
        """
        grid

        grid creates a rectangular grid whose dimensions and resolution is dependent on the res parameter

        Keyword Arguments:
            res {int | slice | tuple[slice,...]} -- the resolution and dimensions of the grid passed as n slices (default: {slice(0,1,200)})

        Returns:
            torch.Tensor -- {d1,...,dn,n} tensor with n+1 dimensions where the last dimension is the coordinates and therefore of size n. all other dimensions have sizes coresponding to the resolution specified by their coresponding res parameter
        """
        if isinstance(res, int):
            res = (slice(0, 1, res),)
        elif isinstance(res, slice):
            res = (res,)
        axes = [torch.linspace(r.start, r.stop, r.step) for r in res]
        meshgrid = torch.meshgrid(axes, indexing="ij")
        return torch.stack(meshgrid, dim=-1)

    def resize_modes(
        self, target_modes: int | tuple[int, ...] | Self, rescale: bool = True
    ):
        """
        resize_modes

        creates a copy of this basis with modes resized to target modes

        Arguments:
            target_modes {int | tuple[int, ...] | Basis} -- the target mode or basis mode to resize this basis to


        Returns:
            Basis -- A copy of this basis with resized coefficients
        """
        target_basis = None
        if isinstance(target_modes, int):
            target_modes = (target_modes,)
        elif isinstance(target_modes, Basis):
            target_basis = target_modes
            target_modes = target_basis.modes

        copy = self.copy()
        coeff = copy.coeff
        if self.time_dependent:
            coeff = coeff.flatten(0, 1)
            coeff = resize_modes(coeff, target_modes, rescale=rescale)
            coeff = coeff.unflatten(0, (len(self), self.time_size))
            if target_basis is not None:
                # interpolate time dependent coefficients
                index_float = torch.linspace(0, 1, target_basis.time_size) * (
                    self.time_size - 1
                )
                coeff = self.interpolate_time_tensor(coeff, index_float)
        else:
            coeff = resize_modes(coeff, target_modes, rescale=rescale)
        copy.coeff = coeff
        return copy

    def perturb(
        self,
        std_ratio: float = 0.1,
        rand_func=torch.randn,
        generator: torch.Generator | None = None,
    ) -> Self:
        """
        perturb

        add noise to the function values of the basis coefficients

        Keyword Arguments:
            std_ratio {float} -- ratio of noise to standard deviation of function values (default: {0.1})
            rand_func {Callable} -- random value generator function (default: {torch.randn})
            generator {torch.Generator | None} -- generator for reproducability (default: {None})

        Returns:
            Basis -- a perturbed copy of this basis
        """
        values = self.inv_transform(self.coeff)
        perturbed_values = (
            values
            + rand_func(
                values.shape,
                generator=generator,
                dtype=self.coeff.dtype
                if self._complex_funcs
                else self.coeff.real.dtype,
            )
            * std_ratio
            * values.std()
        )
        copy = self.copy()
        copy.coeff = self.transform(perturbed_values)
        return copy

    def __getitem__(self, indices):
        copy = self.copy()
        idx = torch.arange(len(copy))[indices].reshape(-1)
        # ensure the indexing results in list of coefficients (0th dimension is the list index)
        copy.coeff = copy.coeff[idx, ...]
        return copy

    def to_time_dependent(self, nt: int | None = None):
        copy = self.copy()
        if self.time_dependent:
            return copy

        res_modes = tuple(
            slice(0, period, mode)
            for mode, period in zip(self.modes[1:], self.periods[1:], strict=True)
        )
        if nt is None:
            nt = self.modes[0]
        res_modes = (slice(0, self.periods[0], nt), *res_modes)

        val = self.get_values(res=res_modes)
        time_dependent_coeff = self.transform(val.flatten(0, 1)).reshape(
            (len(self), *self.modes)
        )
        copy.time_dependent = True
        copy.coeff = time_dependent_coeff
        return copy

    def to_time_independent(self):
        copy = self.copy()
        if not self.time_dependent:
            return copy

        # since periods combine time period with spatial period, get only the spatial ones with index [1:]
        res_modes = tuple(
            slice(0, period, mode)
            for mode, period in zip(self.modes, self.periods[1:], strict=True)
        )
        res_modes = (slice(0, self.periods[0], self.time_size), *res_modes)
        val = self.get_values(res=res_modes)
        time_dependent_coeff = self.transform(val)
        copy.time_dependent = False
        copy.coeff = time_dependent_coeff
        return copy

    @staticmethod
    def interpolate_time_tensor(x: torch.Tensor, index_float: torch.Tensor):
        x_interp = interpolate_tensor(x, index_float, dim=1)

        return x_interp


BasisSubType = TypeVar("BasisSubType", bound="Basis")
