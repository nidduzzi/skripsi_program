# Fourier / transform conventions

This project touches three DFT conventions. The goal: **our convention is
byte-for-byte `torch.fft`, so no conversion is ever needed between our code and
`torch.fft`.** Only `torchkbnufft` differs, and that conversion is confined to a
single adapter (`basis/_nufft.py`). Nothing else in the codebase should do
convention juggling.

## 1. `torch.fft` (the reference)

- **Forward** (`fft`, `norm="backward"`): `X_k = Σ_n x_n · exp(-2πi·k·n/N)` — no scaling.
- **Inverse** (`ifft`): `x_n = (1/N) Σ_k X_k · exp(+2πi·k·n/N)`.
- **Wavenumber order**: `fftfreq(N)·N = [0, 1, …, ⌈N/2⌉-1, -⌊N/2⌋, …, -1]`
  (DC first, positive freqs, then negative).

## 2. Our `FourierBasis` convention — *identical to `torch.fft`*

- `wave_number(N)` == `fftfreq(N)·N` (same ordering).
- `fn(x, k, period=L)` = `exp(2πi·k·x/L)` (the `+` inverse kernel).
- `transform` == `fft` (forward, `-` sign, no scale). *Verified*:
  `test_fuzz_transform_matches_torch_fft`.
- `inv_transform` / `evaluate` == `ifft` (`+` sign, `1/N`). *Verified*:
  `test_mms_evaluate_matches_inv_transform_on_grid`,
  `test_fuzz_inv_transform_matches_torch_ifft`.
- Domain: period `L` per axis (default 1); the sampling grid is owned by the
  injected `SamplingScheme` (`PeriodicUniform` = half-open `[0,L)`, FFT-eligible;
  `ClosedUniform` = closed `[0,L]`, matmul path).

**Consequence:** anywhere we hold coefficients, they are in fft order with the
standard (backward) normalization. We can hand them straight to `torch.fft`
(and vice-versa) with zero conversion.

## 3. `torchkbnufft` (`KbNufft` / `KbNufftAdjoint`) — the one that differs

- **Image order**: natural / centered (DC at index `N//2`), *not* fft order.
- **Exponent**: `kdata_j = Σ_n image_n · exp(-i·ktraj_j·(n-center))`, `-` sign.
- **`ktraj`**: shape `(ndim, npts)`, radians (nominally `[-π, π]`); it is the
  phase per unit wavenumber index, i.e. `2π·x/L` for physical position `x`.
- **Normalization**: none (bare sum), so divide by `prod(modes)` to match our
  `evaluate`.

### Adapter (fixed mapping, lives only in `basis/_nufft.py`)

To evaluate our (fft-ordered) `coeff` at non-uniform points `x` with period `L`:

1. reorder each coeff axis fft→natural: `index_select(argsort(fftfreq(N)))`;
2. `ktraj = -2π·x/L` per axis (the `-` cancels torchkbnufft's `-` sign, giving
   our `+` inverse kernel);
3. divide the result by `prod(modes)`.

Validated to the KB gridding accuracy (~5e-4 rel) against the exact matmul path
in 1D/2D/batched (`tests/basis/test_nufft.py`).

## Can these be unified?

- **Ours ≡ `torch.fft`** already — nothing to do; treat them as the same.
- **`torchkbnufft`** cannot be changed (external, centered/`-`-sign API), so a
  conversion is unavoidable — but it is a *single fixed adapter* at the library
  boundary (`_nufft.py`), never sprinkled through the code. Any future
  non-uniform-transform backend gets the same treatment: one adapter, convert to
  our (== `torch.fft`) convention at its edge.
