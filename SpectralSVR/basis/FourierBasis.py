from .__base import (
    Basis,
    DomainAxis,
    DomainInputType,
    domainInputType_to_tuple,
    EvaluationModeType,
    ResType,
    transformResType_to_tuple,
)
from .__base import EvaluationStrategy
from .sampling import (
    FourierAcceptableScheme,
    PeriodicUniform,
    SamplingInputType,
    SamplingScheme,
    samplings_to_tuple,
)
from .strategy import DEFAULT_EVALUATION_STRATEGY
from ._nufft import nufft_available, nufft_evaluate
from ..utils import to_complex_coeff
import torch
import math
from typing_extensions import Self, Literal, Callable
from functools import partial


## Fourier basis
class FourierBasis(Basis):
    coeff_dtype = torch.complex64

    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        complex_funcs: bool = False,
        domain: DomainInputType = None,
        time_dependent: bool = False,
        sampling: FourierAcceptableScheme | None = None,
        strategy: EvaluationStrategy | None = None,
    ) -> None:
        super().__init__(
            coeff,
            complex_funcs=complex_funcs,
            time_dependent=time_dependent,
            domain=domain,
            sampling=sampling,
            strategy=strategy,
        )

    @staticmethod
    def default_sampling() -> FourierAcceptableScheme:
        return PeriodicUniform()

    @property
    def coeff(self) -> torch.Tensor:
        return self._coeff

    @coeff.setter
    def coeff(self, coeff: torch.Tensor | None):
        if coeff is not None:
            assert coeff.is_complex(), "the coefficients passed in need to be complex"
        Basis.coeff.fset(self, coeff)  # type: ignore[attr-defined]

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i=0,
        n=0,
        domain: DomainInputType = None,
        **kwargs,
    ) -> torch.Tensor:
        coeff = self.coeff
        assert coeff is not None and coeff.numel() > 0, (
            "basis has no coefficients; set them via the constructor or the coeff "
            "setter before evaluating"
        )
        if domain is None:
            domain = self.domain
        modes = self.get_modes(coeff, time_dependent=self.time_dependent)
        assert modes is not None, (
            "modes is none, set it in the function parameters, at initialization of this basis, or via class properties"
        )
        return self.evaluate(
            coeff=coeff.to(device=x.device),
            x=x,
            t=t,
            domain=domain,
            i=i,
            n=n,
            time_dependent=self.time_dependent,
            strategy=self.strategy,
        )

    @classmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i=0,
        n=0,
        time_dependent: bool = False,
        domain: DomainInputType = None,
        strategy: EvaluationStrategy | None = None,
        **kwargs,
    ) -> torch.Tensor:
        strategy = strategy if strategy is not None else DEFAULT_EVALUATION_STRATEGY
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        if n > 0:
            init_len = len(coeff)
            coeff = coeff[i : i + n]
            assert True if init_len < 1 else len(coeff) > 0, (
                "pass valid values for i and n"
            )
        if not coeff.is_complex():
            coeff = to_complex_coeff(coeff)

        modes = cls.get_modes(coeff, time_dependent=time_dependent)
        if time_dependent:
            assert t is not None, "t must not be none for time dependent evaluations"
            domain_t = domainInputType_to_tuple(domain, (coeff.shape[1:]))
            assert len(domain_t) > 1, (
                f"domain given for time dependent evaluation must at least be of dimension 2, got {len(domain_t)}"
            )
            basis = cls.fn(x, modes, domain=domain_t[1:])
            # evaluate
            sum_coeff_x_basis = cls.sum_mul(coeff.flatten(2), basis)
            # interpolate onto the time axis, mapping t through its domain interval
            t_start, t_stop = domain_t[0]
            index_float = (
                (t.flatten().real - t_start) / (t_stop - t_start) * (coeff.shape[1] - 1)
            )
            sum_coeff_x_basis = cls.interpolate_time_tensor(
                sum_coeff_x_basis, index_float
            )

        else:
            npoints = x.shape[0]
            dense_elems = npoints * math.prod(modes)
            itemsize = coeff.element_size()
            domain_tuple = domainInputType_to_tuple(domain, modes)
            if strategy.use_approximate(dense_elems, itemsize) and nufft_available():
                # NUFFT already includes the 1/prod(modes) scaling
                return nufft_evaluate(coeff, x, domain_tuple)
            sum_coeff_x_basis = cls._dense_evaluate(
                coeff, x, modes, domain_tuple, strategy
            )

        scaling = 1.0 / torch.prod(torch.Tensor(modes))
        return scaling * sum_coeff_x_basis

    @classmethod
    def _dense_evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        modes: tuple[int, ...],
        domain: tuple[DomainAxis, ...],
        strategy: EvaluationStrategy,
    ) -> torch.Tensor:
        """Exact ``sum_k coeff_k exp(2*pi*i*k*(x-a)/L)`` over points ``x``.

        Uses the *separable* structure of the Fourier kernel:
        ``exp(2*pi*i*sum_d k_d x_d/L_d) = prod_d exp(2*pi*i*k_d x_d/L_d)``.
        So instead of building the full (npoints * prod(modes)) basis matrix and
        evaluating that many exponentials, it builds one (npoints * m_d) factor
        per axis (only ``sum_d npoints*m_d`` exponentials) and contracts them
        against the coefficients with a single einsum -- far fewer transcendental
        evaluations and no giant intermediate for ndim >= 2. Chunked over points
        so peak memory stays O(chunk * max_d(m_d)). Exact (no gridding).

        Shapes:
            coeff   -- (batch, *modes) complex spectrum in fft order.
            x       -- (npoints, ndim) evaluation points (ndim == len(modes)).
            modes   -- tuple of length ndim.
            returns -- (batch, npoints) complex, the unscaled sum (evaluate()
                       applies the 1/prod(modes) factor).
        """
        assert x.shape[1] == len(modes), (
            f"x has {x.shape[1]} dims but modes has {len(modes)}"
        )
        domain_t = domainInputType_to_tuple(domain, modes)
        ndim = len(modes)
        mode_syms = "abcdefghijklmn"[:ndim]  # one contraction index per axis
        eq = (
            f"z{mode_syms},"
            + ",".join(f"y{s}" for s in mode_syms)
            + "->zy"  # (batch, modes...) x (points, m_d) per axis -> (batch, points)
        )
        # chunk points so each per-axis factor fits the memory budget; the
        # per-point cost of the separable path is the largest single axis (m_d)
        chunk = strategy.chunk_points(max(modes), coeff.element_size())
        blocks = []
        for start in range(0, x.shape[0], chunk):
            xb = x[start : start + chunk]
            factors = [
                torch.exp(
                    2j
                    * torch.pi
                    * (xb[:, d].unsqueeze(-1).to(coeff) - domain_t[d][0])
                    * cls.wave_number(m).flatten().to(coeff)
                    / (domain_t[d][1] - domain_t[d][0])
                )
                for d, m in enumerate(modes)
            ]
            blocks.append(torch.einsum(eq, coeff, *factors))
        return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)

    @staticmethod
    def sum_mul(coeff_flat: torch.Tensor, basis: torch.Tensor):
        sum_coeff_x_basis = coeff_flat.matmul(basis.movedim(0, -1).flatten(0, -2))
        return sum_coeff_x_basis

    @staticmethod
    def fn(
        x: torch.Tensor,
        modes: int | tuple[int, ...] | None = None,
        domain: DomainInputType = None,
        constant=2j * torch.pi,
        transpose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if modes is None:
            raise ValueError("modes should not be None. It is required.")
        if isinstance(modes, int):
            modes = (modes,)
        domain = domainInputType_to_tuple(domain, modes)

        assert len(x.shape) > 1, (
            "x must have at least 2 dimensions, the format needs to be row of points, the first dimension of the tensor being each row and the second being dimensions of the points"
        )
        assert x.shape[0] > 0, (
            f"x has shape {x.shape}, make sure the first dimension isn't empty ie. has at least one row of samples"
        )
        assert x.shape[1] == len(modes), (
            f"x has dimensions {x.shape[1]} and modes has dimensions {len(modes)}, both need to have the same dimensions (modes specify how many modes in each dimension of the fourier series)"
        )
        assert x.shape[1] == len(domain), (
            f"x has dimensions {x.shape[1]} and domain has dimensions {len(domain)}, both need to have the same dimensions (domain is the (start, stop) interval in each dimension)"
        )
        ndims = x.shape[1]
        if not x.is_floating_point() and not x.is_complex():
            x = x.float()

        # Compute the Fourier basis functions
        # one time for each dimension

        dim = 0
        num_modes = modes[dim]
        dim_basis_shape = [1 for i in range(ndims + 1)]
        dim_basis_shape[0] = x.shape[0]
        dim_basis_shape[dim + 1] = num_modes
        kx = torch.zeros((x.shape[0], modes[0])).reshape(dim_basis_shape).to(x)

        for dim, num_modes in enumerate(modes):
            k = FourierBasis.wave_number(num_modes, target=x).T
            dim_basis_shape = [1 for i in range(ndims + 1)]
            dim_basis_shape[0] = x.shape[0]
            dim_basis_shape[dim + 1] = num_modes
            start_d, stop_d = domain[dim]
            dim_x = (x[:, dim : dim + 1] - start_d).div(stop_d - start_d)
            dim_kx = torch.mm(dim_x, k)
            if transpose:
                dim_kx = dim_kx.T
            dim_kx = dim_kx.reshape(dim_basis_shape)

            kx = kx + dim_kx
        kx = kx.mul(constant).exp_()
        basis = kx

        return basis

    @staticmethod
    def wave_number(modes: int, target: torch.dtype | torch.Tensor = torch.float):
        N = (modes - 1) // 2 + 1
        n = modes // 2
        k1 = torch.arange(0, N)
        k2 = torch.arange(-n, 0)
        k = torch.concat([k1, k2], dim=0).unsqueeze(-1).to(target)
        return k

    @staticmethod
    def derivative_eigenvalues(
        modes: int, length: float, ord: int = 1
    ) -> torch.Tensor:
        # differentiation is diagonal in Fourier space: multiply by (2*pi*i*k/L)
        k = FourierBasis.wave_number(modes).flatten()
        return (2 * torch.pi * 1j * k / length) ** ord

    @staticmethod
    def dealias_mask(modes: int, fraction: float = 2.0 / 3.0) -> torch.Tensor:
        # Orszag 2/3 rule by default: keep |k| <= fraction * (modes / 2)
        k = FourierBasis.wave_number(modes).flatten()
        return k.abs() <= fraction * (modes / 2)

    @classmethod
    def generate(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        complex_funcs: bool = False,
        domain: DomainInputType = None,
        value_type: Literal["random", "zero"] = "random",
        **kwargs,
    ) -> Self:
        match value_type:
            case "random":
                coeff = cls.generate_coeff(
                    n,
                    modes,
                    generator=generator,
                    random_func=random_func,
                    complex_funcs=complex_funcs,
                )
            case "zero":
                coeff = cls.generate_empty(n, modes)
            case _:
                raise RuntimeError(
                    f"FourierBasis generate expects value_type random or zero but got {value_type}"
                )

        return cls(
            coeff,
            domain=domain,
            complex_funcs=complex_funcs,
        )

    @classmethod
    def generate_empty(
        cls,
        n: int,
        modes: int | tuple[int, ...],
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = (modes,)
        n_modes = len(modes)
        assert n_modes > 0, "modes should have at least one dimension"
        coeff = torch.zeros((n, *modes), dtype=torch.complex64)
        return coeff

    @classmethod
    def generate_coeff(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func: Callable[..., torch.Tensor] = torch.randn,
        complex_funcs: bool = False,
        scale: bool = True,
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = (modes,)
        n_modes = len(modes)
        assert n_modes > 0, "modes should have at least one element"
        random_func = partial(random_func, generator=generator)
        coeff = random_func((n, *modes), dtype=torch.complex64)
        if not complex_funcs:
            # Real-valued functions need a conjugate-symmetric spectrum.
            coeff = cls.enforce_hermitian(coeff)
        if scale:
            # Heuristic amplitude so generated functions have O(1) magnitude
            # regardless of mode count (the inverse DFT sums prod(modes) terms).
            scaler = torch.tensor(modes).sum() * 0.2
            coeff = coeff.mul(scaler)
        return coeff

    @staticmethod
    def enforce_hermitian(coeff: torch.Tensor) -> torch.Tensor:
        """Project a spectrum onto Hermitian symmetry so its inverse is real.

        A real signal has a conjugate-symmetric spectrum, ``X[k] = conj(X[-k])``.
        In fft ordering the ``-k`` reindex of an axis is ``flip`` then ``roll(1)``
        (0->0, 1->N-1, 2->N-2, ...). Averaging with the conjugate of that
        frequency-mirrored spectrum enforces the symmetry over every mode axis
        (axis 0 is the sample axis and is left untouched); DC and Nyquist map to
        themselves and become real.
        """
        mirror = coeff
        for dim in range(1, coeff.ndim):
            mirror = torch.flip(mirror, (dim,)).roll(1, dim)
        return 0.5 * (coeff + mirror.conj())

    @staticmethod
    def _raw_transform(
        f: torch.Tensor,
        func: Literal["forward", "inverse"],
        res: slice,
        sampling: SamplingScheme,
        domain: DomainAxis,
        strategy: EvaluationStrategy,
    ) -> torch.Tensor:
        assert torch.is_complex(f), (
            "f is not complex, cast it to complex first eg. f + 0j"
        )
        match func:
            case "forward":
                sign = -1
            case "inverse":
                sign = 1
        mode = f.shape[1]
        start, stop = domain
        # the FFT is only valid on the full periodic grid over the domain at the
        # native resolution; a different res.step or a partial window would
        # alias, so fall back to the explicit basis matrix in that case. (The DFT
        # is translation-agnostic, so a nonzero start is fine as long as the grid
        # spans the whole domain.)
        can_use_fft = (
            res.start == start
            and res.stop == stop
            and sampling.is_periodic
            and sampling.supports_fft
            and mode == res.step
        )
        if can_use_fft:
            if func == "forward":
                F = torch.fft.fft(f, dim=1, n=res.step, norm="backward")
            elif func == "inverse":
                F = torch.fft.ifft(f, dim=1, n=res.step, norm="forward")
        else:
            # the sampling scheme owns node placement over the domain (real
            # coordinates -- do not cast to the complex coeff dtype)
            n = sampling.nodes(res.step, res.start, res.stop).to(device=f.device)
            # inverse (coeff -> values at nodes) is a type-2 NUFFT, so route the
            # large non-uniform case through it instead of the dense (npts*mode)
            # basis matrix. (The forward/type-1 adjoint would need its own
            # calibration; it stays on the exact matmul for now.)
            if (
                func == "inverse"
                and strategy.use_approximate(res.step * mode, f.element_size())
                and nufft_available()
            ):
                # nufft_evaluate divides by prod(modes)=mode; _raw_transform
                # returns the unscaled sum (inv_transform applies 1/N later)
                return nufft_evaluate(f, n.view(-1, 1), (domain,)) * mode
            e = FourierBasis.fn(
                n.to(f).view(-1, 1),  # match coeff dtype for the matmul
                mode,
                domain=domain,
                constant=sign * 2j * torch.pi,
            )
            F = torch.mm(f, e.T)

        assert isinstance(F, torch.Tensor), (
            f"Something went wrong during the raw transform, expected result of {torch.Tensor}, but got {type(F)}"
        )

        return F

    @staticmethod
    def _ndim_transform(
        f: torch.Tensor,
        dim: int,
        func: Literal["forward", "inverse"],
        res: slice,
        sampling: SamplingScheme,
        domain: DomainAxis,
        strategy: EvaluationStrategy,
    ) -> torch.Tensor:
        # flatten so that each extra dimension is treated as a separate "sample"
        # move dimension to transform to the end so that it can stay intact after f is flatened
        f_transposed = f.moveaxis(dim, -1)
        # flatten so that the last dimension is intact
        f_flatened = f_transposed.flatten(0, -2)

        F_flattened = FourierBasis._raw_transform(
            f_flatened,
            func=func,
            res=res,
            sampling=sampling,
            domain=domain,
            strategy=strategy,
        )
        # unflatten so that the correct shape is returned
        F_transposed = F_flattened.reshape((*f_transposed.shape[:-1], res.step))
        F = F_transposed.moveaxis(-1, dim)

        return F

    @staticmethod
    def transform(
        f: torch.Tensor,
        res: ResType | None = None,
        sampling: SamplingInputType = None,
        strategy: EvaluationStrategy | None = None,
        domain: DomainInputType = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        transform

        Discrete Fourier transform of complex-valued signal f.

        Arguments:
            f {torch.Tensor} -- m discretized functions (first dim is samples)
            res {tuple[slice,...] | None} -- resolution/bounds the function was
                evaluated at (default: same as f, bounds = the domain).
            sampling {FourierAcceptableScheme | None} -- how f was sampled;
                determines the grid periodicity and FFT eligibility (default:
                PeriodicUniform, i.e. periodic + FFT).
            domain -- per-axis (start, stop) interval (default: (0, 1)).

        Returns:
            torch.Tensor -- m complex valued coefficients of f
        """
        strategy = strategy if strategy is not None else DEFAULT_EVALUATION_STRATEGY
        ndims = len(f.shape)
        assert ndims >= 2, (
            f"f has shape {f.shape}, It needs to have at least two dimensions with the first being m samples"
        )
        if not torch.is_complex(f):
            f = f * (1 + 0j)
        domain = domainInputType_to_tuple(domain, f.shape[1:])
        sampling_axes = samplings_to_tuple(sampling, ndims - 1, PeriodicUniform())
        # Res should by default span the domain, not the unit interval
        res = transformResType_to_tuple(res, tuple(f.shape[1:]), domain)
        # perform 1d transform over every dimension
        F = f
        for cdim in range(1, ndims):
            F = FourierBasis._ndim_transform(
                F,
                dim=cdim,
                func="forward",
                res=res[cdim - 1],
                sampling=sampling_axes[cdim - 1],
                domain=domain[cdim - 1],
                strategy=strategy,
            )

        return F

    @staticmethod
    def inv_transform(
        f: torch.Tensor,
        res: ResType | None = None,
        sampling: SamplingInputType = None,
        strategy: EvaluationStrategy | None = None,
        scale: bool = True,
        domain: DomainInputType = None,
        **kwargs,
    ):
        """
        inv_transform

        Discrete inverse Fourier transform of coefficients f.

        Arguments:
            f {torch.Tensor} -- m complex coefficient vectors (first dim samples)
            res {tuple[slice,...] | None} -- resolution/bounds to evaluate at
                (default: same as f, bounds = the domain).
            sampling {FourierAcceptableScheme | None} -- evaluation grid scheme;
                determines periodicity and FFT eligibility (default:
                PeriodicUniform).
            scale {bool} -- whether outputs are scaled by N (default: True).
            domain -- per-axis (start, stop) interval (default: (0, 1)).

        Returns:
            torch.Tensor -- m function value vectors
        """
        strategy = strategy if strategy is not None else DEFAULT_EVALUATION_STRATEGY
        ndims = len(f.shape)
        assert ndims >= 2, (
            f"f has shape {f.shape}, It needs to have at least two dimensions with the first being m samples"
        )
        if not torch.is_complex(f):
            f = f * (1 + 0j)
        # inverse-DFT normalisation is 1/N over the MODE count; capture it before
        # the loop rewrites the shape to the (possibly larger) evaluation
        # resolution. Dividing by the resolution instead would scale values by
        # (modes/res)^ndim whenever res != modes (e.g. plotting, residual grids).
        modes = tuple(f.shape[1:])
        domain = domainInputType_to_tuple(domain, f.shape[1:])
        sampling_axes = samplings_to_tuple(sampling, ndims - 1, PeriodicUniform())
        # Res should by default span the domain, not the unit interval
        res = transformResType_to_tuple(res, modes, domain)

        # perform 1d transform over every dimension
        for cdim in range(1, ndims):
            f = FourierBasis._ndim_transform(
                f,
                dim=cdim,
                func="inverse",
                res=res[cdim - 1],
                sampling=sampling_axes[cdim - 1],
                domain=domain[cdim - 1],
                strategy=strategy,
            )

        if scale:
            f = f.div(torch.tensor(modes).prod())
        return f

    def _diff_multiplier(self, dim: int, ord: int) -> torch.Tensor:
        """Per-mode Fourier differentiation multiplier ``(2*pi*i*k / length)^ord``.

        Broadcasts over the coefficient tensor along ``dim``. ``grad`` multiplies
        by it; ``integral`` divides by it.
        """
        if self.time_dependent:
            # disregard the (sample-like) time dimension for spatial derivatives
            dim = dim - 1
        eig = self.derivative_eigenvalues(self.modes[dim], self.lengths[dim], ord)
        multiplier_dims = tuple(
            1 if i != dim else self.modes[i] for i in range(self.ndim)
        )
        if self.time_dependent:
            multiplier_dims = (1, *multiplier_dims)
        return eig.reshape(multiplier_dims).to(self.coeff)

    def _finite_diff_time(self, op: Literal["grad", "integral"], ord: int) -> Self:
        """Finite-difference derivative/antiderivative along the time samples."""
        copy = self.copy()
        dt = self.lengths[0] / (self.time_size - 1)
        coeff = copy.coeff
        for _ in range(ord):
            if op == "grad":
                coeff = torch.gradient(coeff, spacing=dt, dim=1)[0]
            else:
                coeff = coeff.cumsum(1).mul(dt)
        copy.coeff = coeff
        return copy

    def grad(self, dim: int = 0, ord: int = 1) -> Self:
        if dim == 0 and self.time_dependent:
            return self._finite_diff_time("grad", ord)
        copy = self.copy()
        # the k=0 mode along `dim` is already zeroed by its eigenvalue
        # (2*pi*i*0/L)^ord = 0, so no extra DC handling is needed (and zeroing a
        # fixed last-axis DC would wrongly wipe a non-`dim` axis for ndim > 1).
        copy.coeff = copy.coeff.mul(self._diff_multiplier(dim, ord))
        return copy

    def integral(self, dim: int = 0, ord: int = 1) -> Self:
        if dim == 0 and self.time_dependent:
            return self._finite_diff_time("integral", ord)
        copy = self.copy()
        multiplier = self._diff_multiplier(dim, ord)
        coeff = copy.coeff.div(multiplier)
        # dividing by the zero eigenvalue at `dim`'s k=0 gives inf/nan; drop that
        # mode (the integration constant) along the correct axis, not the last.
        coeff = torch.where(multiplier == 0, torch.zeros_like(coeff), coeff)
        copy.coeff = coeff
        return copy

    def copy(self) -> Self:
        basis_copy = super().copy()
        return basis_copy

    def plot_coefficients(
        self,
        i: int = 0,
        n: int = 1,
        plt=None,
        component: Literal["magnitude", "real", "imag"] = "magnitude",
        legend: bool = True,
        **kwargs,
    ):
        """Plot the Fourier spectrum: coefficient ``component`` versus wavenumber.

        1D bases plot against (frequency-sorted) wavenumbers; 2D bases show the
        (fft-shifted) coefficient grid via imshow.
        """
        from ._plot import plot_fourier_coefficients

        return plot_fourier_coefficients(
            self, i=i, n=n, plt=plt, component=component, legend=legend, **kwargs
        )

    @staticmethod
    def prefered_evaluation_mode() -> EvaluationModeType:
        return "inverse transform"
