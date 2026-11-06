# Changelog

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
