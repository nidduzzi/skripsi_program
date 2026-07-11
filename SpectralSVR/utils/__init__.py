import torch
import logging
import typing
from typing import Callable, TypedDict, cast, overload
from torchdiffeq import odeint
from functools import partial
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    symmetric_mean_absolute_percentage_error,
    r2_score,
    relative_squared_error,
)

logger = logging.getLogger(__name__)

Number = int | float

DeviceInput = torch.device | str | None


def resolve_device(device: DeviceInput = None) -> torch.device:
    """Resolve a device, defaulting to CUDA when available else CPU.

    Centralizes device selection so it can be injected/overridden in one place
    instead of duplicating ``cuda if available`` checks across the codebase.
    """
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_complex_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_complex_coeff

    convert a tensor containing rows of real coefficient vectors into rows of complex coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p tensor where p is twice the number modes

    Returns:
        torch.Tensor -- m by p/2 tensor of elements in complex numbers
    """
    if torch.is_complex(coeff):
        logger.debug("coeff is already complex")
        return coeff
    converted_coeff = torch.view_as_complex(coeff.reshape((*coeff.shape[:-1], -1, 2)))
    return converted_coeff


def to_mag_angle(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_mag_angle

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.debug("coeff is already real")
        return coeff
    converted_coeff = torch.empty(
        (coeff.shape[0], coeff.shape[1] * 2), dtype=coeff.real.dtype
    )
    converted_coeff[:, ::2] = coeff.abs()
    converted_coeff[:, 1::2] = coeff.angle()

    return converted_coeff


def to_real_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_real_coeff

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.debug("coeff is already real")
        return coeff
    converted_coeff = torch.view_as_real(coeff).flatten(-2)

    return converted_coeff


class StandardScaler:
    def __init__(self):
        pass

    @staticmethod
    def _get_tensor_tuple(xs: tuple[torch.Tensor, ...] | torch.Tensor):
        if isinstance(xs, torch.Tensor):
            return (xs,)
        else:
            return xs

    @staticmethod
    def _get_m(x: torch.Tensor):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        m = x_real.mean(0, keepdim=True)
        return m

    @staticmethod
    def _get_s(x: torch.Tensor, eps: float):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        s = x_real.std(0, unbiased=False, keepdim=True)
        s[s <= eps] = eps
        return s

    def fit(self, xs: tuple[torch.Tensor, ...] | torch.Tensor, eps=1e-12):
        xs = self._get_tensor_tuple(xs)
        self.ms = tuple(self._get_m(x) for x in xs)
        self.ss = tuple(self._get_s(x, eps) for x in xs)
        self.xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        self.xs_dims = tuple(x.shape[1] for x in xs)
        return self

    @staticmethod
    def _translate(x: torch.Tensor, m: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert x.shape[1] == m.shape[1], (
            f"{x_name} needs to have the same 2nd dimension now and at fitting"
        )
        x_translated = x.subtract(m)
        return x_translated

    @staticmethod
    def _scale(x: torch.Tensor, s: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert x.shape[1] == s.shape[1], (
            f"{x_name} needs to have the same 2nd dimension now and at fitting"
        )
        x_scaled = x.div(s)
        return x_scaled

    def _check_consistency(self, xs: tuple[torch.Tensor, ...]):
        xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        assert xs_is_complex == self.xs_is_complex, (
            f"make sure the complex tensors are the same order when fitting, current complex: {xs_is_complex}, fitting complex: {self.xs_is_complex}"
        )
        xs_dims = tuple(x.shape[1] for x in xs)
        assert xs_dims == self.xs_dims, (
            f"make sure the tensors have the same 2nd dimensions with the ones used at fitting, current dims: {xs_dims}, fitting dims: {self.xs_dims}"
        )

    def get_subset_scaler(self, indices: int | list[int]):
        if isinstance(indices, int):
            indices = [indices]
        new_scaler = StandardScaler()
        new_scaler.ms = tuple(self.ms[index] for index in indices)
        new_scaler.ss = tuple(self.ss[index] for index in indices)
        new_scaler.xs_is_complex = tuple(self.xs_is_complex[index] for index in indices)
        new_scaler.xs_dims = tuple(self.xs_dims[index] for index in indices)
        return new_scaler

    @typing.overload
    def transform(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def transform(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def transform(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        _xs = self._get_tensor_tuple(xs)
        self._check_consistency(_xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, _xs, strict=False)
        )
        xs_transformed = tuple(
            self._translate(x, m, i=i)
            for i, (x, m) in enumerate(zip(xs_real, self.ms, strict=False))
        )
        xs_transformed = tuple(
            self._scale(x, s, i=i)
            for i, (x, s) in enumerate(zip(xs_transformed, self.ss, strict=False))
        )
        for x, m in zip(xs_transformed, self.ms, strict=False):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())

        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed, strict=False)
        )
        return xs_out[0] if is_tensor_input else xs_out

    @typing.overload
    def inverse(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def inverse(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def inverse(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        _xs = self._get_tensor_tuple(xs)
        self._check_consistency(_xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, _xs, strict=False)
        )
        xs_transformed = tuple(
            self._scale(x, 1.0 / s, i=i)
            for i, (x, s) in enumerate(zip(xs_real, self.ss, strict=False))
        )
        xs_transformed = tuple(
            self._translate(x, -1.0 * m, i=i)
            for i, (x, m) in enumerate(zip(xs_transformed, self.ms, strict=False))
        )
        for x, m in zip(xs_transformed, self.ms, strict=False):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())
        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed, strict=False)
        )
        return xs_out[0] if is_tensor_input else xs_out

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        scaler: StandardScaler = torch.load(path)
        assert isinstance(scaler, StandardScaler), (
            "Loaded object is not a valid instance of StandardScaler"
        )
        return scaler


def scale_to_standard(x: torch.Tensor):
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _safe_r2(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # r2_score is undefined for a single sample.
    if targets.shape[0] > 1:
        return r2_score(preds, targets)
    return torch.tensor(float("nan"))


# Default metric set; inject a different mapping into get_metrics to override.
# TODO: trace problem of rse and rrse values blowing up when a model trained
# with noise is tested against the clean version of u_coeff_targets
DEFAULT_METRICS: dict[str, MetricFn] = {
    "mse": mean_squared_error,
    "rmse": partial(mean_squared_error, squared=False),
    "mae": mean_absolute_error,
    "r2": _safe_r2,
    "smape": symmetric_mean_absolute_percentage_error,
    "rse": relative_squared_error,
    "rrse": partial(relative_squared_error, squared=False),
}


class Metrics(TypedDict):
    """Return type of ``get_metrics`` with the default metric set."""

    mse: float
    rmse: float
    mae: float
    r2: float
    smape: float
    rse: float
    rrse: float
    pred_nan_sum: int


@overload
def get_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Metrics: ...
@overload
def get_metrics(
    preds: torch.Tensor, targets: torch.Tensor, metrics: dict[str, MetricFn]
) -> dict[str, float]: ...


def get_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, MetricFn] | None = None,
) -> Metrics | dict[str, float]:
    active = DEFAULT_METRICS if metrics is None else metrics
    results = {name: fn(preds, targets).item() for name, fn in active.items()}
    results["pred_nan_sum"] = int(torch.isnan(preds).sum().item())
    # Default set matches the Metrics keys exactly; a custom set does not.
    return cast(Metrics, results) if metrics is None else results


def resize_modes(x: torch.Tensor, target_modes: int | tuple[int, ...], rescale=True):
    current_modes = tuple(x.shape[1:])
    ncurrent_modes = len(current_modes)
    if isinstance(target_modes, int):
        target_modes = (target_modes,) * ncurrent_modes
    assert ncurrent_modes == len(target_modes), (
        f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {target_modes}"
    )
    x_resized = x
    for dim, (target_mode, current_mode) in enumerate(
        zip(target_modes, current_modes, strict=False), 1
    ):
        device = x.device
        if target_mode < current_mode:
            start_range = torch.tensor(
                range((target_mode - 1) // 2 + 1), dtype=torch.int
            ).to(device=device)
            end_range = torch.tensor(
                range(current_mode - target_mode // 2, current_mode), dtype=torch.int
            ).to(device=device)

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )
        elif target_mode > current_mode:
            start_range = torch.tensor(
                range((current_mode - 1) // 2 + 1), dtype=torch.int
            ).to(device=device)
            # make sure that end range is empty if the coefficient is only size 1
            end_range = torch.tensor(
                range(current_mode // 2, current_mode)
                if current_mode > 1
                else range(0),
                dtype=torch.int,
            ).to(device=device)
            padding_size = target_mode - current_mode
            modes = list(x_resized.shape)
            modes[dim] = padding_size
            padding = torch.zeros(modes).to(x_resized)

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    padding,
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )

    if rescale:
        modes = list(x.shape[1:])
        x_resized = (
            x_resized * torch.tensor(target_modes).prod() / torch.tensor(modes).prod()
        )

    return x_resized


def zero_coeff(x: torch.Tensor, zeroed_modes: int | list[int]):
    if isinstance(zeroed_modes, int):
        zeroed_modes = [zeroed_modes] * len(x.shape[1:])
    assert len(x.shape[1:]) == len(zeroed_modes), (
        f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {zeroed_modes}"
    )
    x_zeroed = x.clone()
    for dim, max_mode in enumerate(zeroed_modes, 1):
        dim_len = x.shape[dim]
        zero_range = torch.tensor(
            range((max_mode - 1) // 2 + 1, dim_len - max_mode // 2)
        )
        x_zeroed.index_fill_(dim, zero_range, torch.zeros((1,), dtype=x.dtype)[0])

    return x_zeroed


def interpolate_tensor(x: torch.Tensor, index_float: torch.Tensor, dim: int = 1):
    index_floor = index_float.floor().to(torch.int)
    index_ceil = index_float.ceil().to(torch.int)
    if index_float.remainder(1).eq(0).all():
        x_interp = x.index_select(dim, index_floor)
    else:
        x_ceil = x.index_select(dim, index_ceil)
        x_floor = x.index_select(dim, index_floor)
        # interpolate coefficients
        index_shape = [1 for _ in range(x_floor.ndim)]
        index_shape[1] = -1
        index_scaler = (
            ((index_float - index_floor) / (index_ceil - index_floor))
            .reshape(index_shape)
            .nan_to_num()
        )
        # ynt + scaler * (ynt1 - ynt)
        # (1 - scaler) * ynt + scaler * ynt1
        x_interp = torch.lerp(x_floor, x_ceil, index_scaler.to(x))

    return x_interp


# SOLVERS

RHSFuncType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SolverSignatureType = Callable[[RHSFuncType, torch.Tensor, torch.Tensor], torch.Tensor]
MixedRHSFuncType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MixedSolverSignatureType = Callable[
    [MixedRHSFuncType, torch.Tensor, torch.Tensor], torch.Tensor
]


def euler_solver(
    rhs_func: RHSFuncType,
    y0: torch.Tensor,
    t: torch.Tensor,
):
    assert len(t.shape) == 1, (
        "t should be a one dimensional tensor of all time evaluation points"
    )
    assert len(t) > 1, "t should have more than one evaluation points"

    solution = torch.zeros((len(t), *y0.shape)).to(y0)

    j = 1
    solution[j - 1] = y0
    for t0, t1 in zip(t[:-1], t[1:], strict=False):
        dt = t1 - t0
        y = solution[j - 1]
        solution[j] = y + dt * rhs_func(t0, y)
        assert solution.isnan().sum() == 0, (
            f"solver encountered nan at timestep {j} (t={t0})"
        )
        # print(f"j {j}")
        j = j + 1
    # print(f"j_last {j}")
    return solution


ExpIntNonlinearType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# ETDRK4 scheme and the complex contour-integral evaluation of its weight
# functions follow:
#   - A.-K. Kassam & L. N. Trefethen, "Fourth-order time-stepping for stiff
#     PDEs", SIAM J. Sci. Comput. 26(4), 2005.
#   - S. M. Cox & P. C. Matthews, "Exponential time differencing for stiff
#     systems", J. Comput. Phys. 176(2), 2002.
# Reference implementations / benchmarks consulted:
#   - https://github.com/gurcani/etdrk4cp
#   - https://github.com/whalenpt/rkstiff
#   - https://docs.sciml.ai/SciMLBenchmarksOutput/dev/SimpleHandwrittenPDE/burgers_spectral_wpd/
#   - ETDRK4 review preprint (preprints.org), provided by the project owner.


def etdrk4_solver(
    linear: torch.Tensor,
    nonlinear: ExpIntNonlinearType,
    y0: torch.Tensor,
    t: torch.Tensor,
    contour_points: int = 32,
) -> torch.Tensor:
    """ETDRK4 exponential time-differencing Runge-Kutta integrator.

    Integrates a semilinear system with a *diagonal* linear part::

        dv/dt = L * v + N(t, v)

    where ``L`` is stiff (e.g. ``-nu |k|^2`` for viscous Burgers in Fourier
    space). The stiff linear part is handled exactly via matrix exponentials;
    the ETDRK4 weight functions are evaluated with the Kassam-Trefethen
    complex contour-integral trick to avoid catastrophic float cancellation
    near ``L = 0``. This is why a generic ``torchdiffeq`` stepper struggles on
    this stiff problem while ETDRK4 stays stable.

    The scheme is applied elementwise over the mode grid, so it works for any
    spatial dimensionality: ``linear`` carries the mode grid shape and the
    solver broadcasts it over the leading sample axis of ``y0``.

    Arguments:
        linear {torch.Tensor} -- (*modes) eigenvalues of the diagonal linear
            operator in the transformed (spectral) space; any mode-grid shape
            (1D, 2D, ...).
        nonlinear {Callable[[t, v], Nv]} -- nonlinear term in the same spectral
            space; receives the current time and state ``v`` of shape
            (samples, *modes) and returns ``N`` of the same shape.
        y0 {torch.Tensor} -- (samples, *modes) initial state in spectral space.
        t {torch.Tensor} -- (steps,) uniformly spaced time points.

    Keyword Arguments:
        contour_points {int} -- number of points on the unit circle used for
            the contour integral (default: {32}).

    Returns:
        torch.Tensor -- (steps, samples, *modes) solution in spectral space.
    """
    assert t.ndim == 1 and len(t) > 1, "t must be a 1D tensor with >1 points"
    assert y0.shape[1:] == linear.shape, (
        f"linear {tuple(linear.shape)} must match y0 mode grid {tuple(y0.shape[1:])}"
    )
    device = y0.device
    h = (t[1] - t[0]).item()

    # Coefficients computed in double precision for accuracy, then cast to y0.
    lin = linear.to(device=device, dtype=torch.complex128)
    E = torch.exp(h * lin)
    E2 = torch.exp(h * lin / 2)

    m = contour_points
    j = torch.arange(1, m + 1, device=device, dtype=torch.float64)
    r = torch.exp(1j * torch.pi * (j - 0.5) / m)  # (m,) on the unit circle
    lr = h * lin.reshape(-1).unsqueeze(-1) + r.unsqueeze(0)  # (prod(modes), m)
    shape = lin.shape

    def _coef(expr: torch.Tensor) -> torch.Tensor:
        return (h * expr.mean(dim=-1).real).reshape(shape)

    Q = _coef((torch.exp(lr / 2) - 1) / lr)
    f1 = _coef((-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3)
    f2 = _coef((2 + lr + torch.exp(lr) * (-2 + lr)) / lr**3)
    f3 = _coef((-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / lr**3)

    cdtype = y0.dtype if y0.is_complex() else torch.complex64
    rdtype = torch.tensor([], dtype=cdtype).real.dtype
    E, E2 = E.to(cdtype), E2.to(cdtype)
    Q, f1, f2, f3 = (c.to(rdtype) for c in (Q, f1, f2, f3))

    v = y0.to(cdtype)
    solution = torch.empty((len(t), *y0.shape), dtype=cdtype, device=device)
    solution[0] = v
    for i in range(1, len(t)):
        ti = t[i - 1]
        nv = nonlinear(ti, v)
        a = E2 * v + Q * nv
        na = nonlinear(ti, a)
        b = E2 * v + Q * na
        nb = nonlinear(ti, b)
        c = E2 * a + Q * (2 * nb - nv)
        nc = nonlinear(ti, c)
        v = E * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
        solution[i] = v
    return solution


implicit_adams_solver: SolverSignatureType = partial(
    odeint, method="implicit_adams", options={"max_iters": 4}
)

lsoda_solver: SolverSignatureType = partial(
    odeint, method="scipy_solver", options={"solver": "LSODA"}
)
