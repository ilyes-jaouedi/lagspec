# Changelog

## 0.3.0

- Calibrated manifolds (manifold separation): `Manifold`, steering vectors a(u) = G h(u)
  on spatial harmonics, accepted by every method; `Manifold.fit` from a measured table
  in one linear solve (per-direction scale free), `Manifold.off_model`.
- IAA normalises by |a(u)|² per direction (needed on a manifold).
- Array API: CuPy, PyTorch and other array-API libraries through `array-api-compat`
  (optional extra `array-api`); results keep the input's namespace, device and
  precision. The FFT path scatters by pad, fold and roll (no indexed assignment).
  Tested on array-api-strict and PyTorch.

## 0.2.0

- One-axis forward and adjoint transforms shared by every form (`lagspec.transform`).
- Linear forms a(u)ᴴ y: `Lattice.linear`.
- Model covariance from a spectrum, Σ_k p_k a(u_k) a(u_k)ᴴ, through the lags:
  `Lattice.covariance_from_spectrum`.
- IAA: `lagspec.iaa`.
- Chirp-z evaluation path for fine grids on large arrays (`method="czt"`, chosen by
  `"auto"` from a measured crossover).
- Benchmarks for the chirp-z crossover and for IAA.

## 0.1.0

- Lag-domain engine for a(u)ᴴ M a(u) on lattices of any dimension, batched, with
  per-element gains; matrix-product and FFT paths.
- Bartlett, Capon (diagonal loading) and MUSIC.
