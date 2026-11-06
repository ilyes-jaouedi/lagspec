# lagspec

Fast covariance-based spatial spectra for arrays whose elements sit on a lattice.

Bartlett, Capon (MVDR) and MUSIC all evaluate a quadratic form a(u)ᴴ M a(u) on a
grid of directions u, with M = R, R⁻¹ or the noise-subspace projector. When the element
positions are integer multiples of a common step (a uniform linear or rectangular array,
a MIMO virtual array, a sparse or nested array), that quadratic form is a Fourier series
over the **difference co-array**:

```
a(u)ᴴ M a(u) = Σ_l L(l) exp(j κ l·u),   L(l) = Σ_{p_n − p_m = l} M[m, n]
```

`lagspec` computes the lag sums L once per matrix and evaluates the series on the whole
grid with a few small matrix products, an FFT, or a chirp-z transform for a zoom. The
result is **exact** on any grid uniform in direction cosines, and it costs about as much
as a single beam scan instead of N² operations per grid point.

## Status

v0.2:

- the lag-domain engine: 1-D, 2-D and 3-D lattices, batches of matrices, per-element
  calibration gains, and three exact evaluation paths (matrix product, FFT on a full
  period, chirp-z for a zoom), chosen automatically from measured crossovers;
- quadratic forms a(u)ᴴ M a(u), linear forms a(u)ᴴ y, and the reverse operation
  Σ_k p_k a(u_k) a(u_k)ᴴ (model covariance from a spectrum);
- Bartlett, Capon with diagonal loading, MUSIC, and IAA;
- calibrated manifolds: steering vectors expanded on spatial harmonics, fitted from a
  measured table in one linear solve, with every method still on the fast engine;
- array API: NumPy by default, and CuPy, PyTorch or any array-API library through
  `array-api-compat` (results stay on the input's device, in its precision); tested on
  a CUDA GPU (Tesla V100) with PyTorch and CuPy.

Every result is tested against the explicit steering-vector computation.

| Benchmark (`benchmarks/`) | Speed-up vs explicit steering vectors |
|---|---|
| a(u)ᴴ M a(u), 4 × 4 array, 128 matrices, 61 × 61 grid | about 200–400× |
| a(u)ᴴ M a(u), 256-element line, 4096-point grid | about 2000× |
| IAA, 4 × 4 array, 32 matrices, 61 × 61 grid, 10 iterations | about 100× |

Planned: amplitude Capon / APES with spatial smoothing.

## Install

```
pip install git+https://github.com/ilyes-jaouedi/lagspec
```

Requires Python 3.9+ and NumPy. For CuPy, PyTorch or other array-API libraries:

```
pip install "lagspec[array-api] @ git+https://github.com/ilyes-jaouedi/lagspec"
```

## Example

```python
import numpy as np
import lagspec as ls

lat = ls.Lattice.ura(4, 4)                  # 4 x 4 array, half-wavelength lattice
grid = ls.Grid.linspace((-0.8, -0.6), (0.8, 0.6), (161, 121))   # (u, v) direction cosines

x = np.random.randn(128, 16, 40) + 1j * np.random.randn(128, 16, 40)  # 128 range bins
R = ls.covariance(x)                         # (128, 16, 16)

p_capon = ls.capon(R, lat, grid, loading=1e-3)   # (128, 161, 121)
p_music = ls.music(R, lat, grid, n_sources=2)
p_iaa = ls.iaa(R[:8], lat, grid, n_iter=15)    # first 8 range bins
az, el = ls.azel(*grid.mesh())               # angles of the grid points
```

With a measured steering table (rows `table`, directions `points`), a calibrated
manifold replaces the lattice in every method:

```python
man = ls.Manifold.fit(table, points, lat, extra=1)
print(man.off_model(table_test, points_test))      # off-model energy on held-out directions
p_capon = ls.capon(R, man, grid, loading=1e-3)
```

The same calls accept arrays of any array-API library; the work runs where the input
lives:

```python
import torch
R_gpu = torch.as_tensor(R, device="cuda")
p = ls.capon(R_gpu, lat, grid, loading=1e-3)       # torch tensor on the GPU
```

Any matrix or vector can go through the engine directly: `lat.quad(M, grid)`,
`lat.linear(y, grid)`, `lat.covariance_from_spectrum(p, grid)`.

## How it works

See [docs/theory.md](docs/theory.md): the lag sums, why a grid uniform in direction
cosines makes the evaluation exact, the matrix-product, FFT and chirp-z paths,
calibration gains, linear forms, the model covariance and IAA, calibrated manifolds,
and the cost model.

## Prior work

The idea is not new. Fourier evaluation of Capon spectra for uniform data was described
by Musicus (1985), Liu, Li and Li (1998), and Jakobsson, Marple and Stoica (2000, 2-D).
Related fast implementations exist for IAA (Xue, Xu and Li, 2011; Glentis and Jakobsson,
2012) and root-MUSIC (Zhuang, Li and Manikas, 2010). `lagspec` packages the common
structure, the co-array lag sums, as one engine for any lattice geometry and any
quadratic-form spectrum. See [docs/theory.md](docs/theory.md) for the references.

## License

MIT
