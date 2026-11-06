# Theory

## 1. Setting

An array of N elements. Element n sits at the position p_n · d, where d is the lattice
step and **p_n is an integer vector** in 1, 2 or 3 dimensions. For a far-field direction
with direction-cosine vector u (in 2-D, u = (cos el · sin az, sin el)), the steering
vector is

    a_n(u) = g_n exp(j κ p_n·u),   κ = 2π d / λ

with optional complex per-element gains g_n (calibration). For a half-wavelength
lattice, κ = π.

Uniform linear and rectangular arrays, sparse arrays (nested, co-prime) and the virtual
arrays of MIMO radars (each virtual element at p_tx + p_rx) all have this form. Repeated
positions are allowed.

## 2. Quadratic-form spectra

Many spatial spectra evaluate a quadratic form on a grid of directions:

| Spectrum | M | P(u) |
|---|---|---|
| Bartlett (delay and sum) | R | a(u)ᴴ R a(u) |
| Capon / MVDR | (R + δI)⁻¹ | 1 / a(u)ᴴ M a(u) |
| MUSIC | E_n E_nᴴ (noise subspace) | 1 / a(u)ᴴ M a(u) |

Inside their iterations, IAA, SPICE and covariance-fitting methods also evaluate forms
like these. Computed directly, the cost is O(N²) per grid point and per matrix: with K
grid points and B matrices (for example one per range bin), O(B K N²).

## 3. Lag sums

Expanding the form,

    a(u)ᴴ M a(u) = Σ_{m,n} conj(g_m) g_n M[m,n] exp(j κ (p_n − p_m)·u)

The phase depends on the pair (m, n) only through the **lag** l = p_n − p_m, an integer
vector of the difference co-array. Grouping the pairs by lag gives

    a(u)ᴴ M a(u) = Σ_l L(l) exp(j κ l·u),   L(l) = Σ_{p_n − p_m = l} conj(g_m) g_n M[m,n]

This is a finite Fourier series in u. Its coefficients L(l) are computed once per matrix,
in O(N²), with a fixed 0/1 summation matrix, which is batched as a single matrix product.
In D dimensions the lags fill a box of Λ_1 × … × Λ_D values, with Λ_d at most
2·(extent along d) + 1. A 4 × 4 array has 7 × 7 = 49 lags for 256 matrix entries.

**Calibration gains** only rescale the entries of M, so the evaluation stays exact. This
is the same statement as calibrating the data: with whitened data x' = D⁻¹ x, where
D = diag(g), Capon on x' with ideal steering vectors equals Capon on x with the
calibrated ones.

## 4. Evaluation on a grid

**Uniform grids.** On a grid uniform along each axis, u_d[k] = s_d + k h_d, the series
separates by axis:

    exp(j κ l·u) = Π_d exp(j κ l_d u_d)

The lag array is then contracted with one small matrix E_d[l, k] = exp(j κ l (s_d + k h_d))
per axis. In 2-D this is E_1ᵀ L E_2. The cost per matrix is
O(Λ_1 K_1 Λ_2 + K_1 K_2 Λ_2): for small arrays, about one beam scan.

**Periodic grids and the FFT.** If κ h_d K_d = 2π, the grid covers one period of the
array response (u ∈ [−1, 1) for κ = π). Then

    exp(j κ l (s + k h)) = exp(j κ l s) · exp(j 2π (l mod K) k / K)

so the axis is evaluated by scattering the phased lags into K bins (lags that alias onto
the same bin add up, which is still exact) and one inverse FFT of size K. The cost per
axis is O(K log K), independent of the array size. This path pays off for large arrays.
`method="auto"` chooses it when Λ_d > 8 log₂ K_d (`transform.FFT_FACTOR`, set from `benchmarks/bench_quad.py`): below that, the small matrix product is faster.

**Zoom: the chirp-z transform.** On a grid that is not periodic, typically a fine grid
over a small region, one axis is still a chirp-z transform. With
o = o_0 + i, exp(j κ o u_k) factors into exp(j κ o_0 u_k) · exp(j κ i s) · exp(j κ h i k),
and Bluestein's identity i k = (i² + k² − (k − i)²) / 2 turns the sum over i into a
convolution: three FFTs of length L ≥ Λ + K − 1. This pays off for long arrays only.
`method="auto"` chooses it when Λ K > 8 L log₂ L (`transform.CZT_FACTOR`, from
`benchmarks/bench_czt.py`); for small arrays the matrix product stays faster.

**Exactness.** All three paths are exact, not interpolations: the result equals the explicit
steering-vector computation to rounding error. The tests check this for linear,
rectangular, MIMO, sparse and 3-D lattices.

**Uniform in direction cosines, not in angle.** An angle grid uniform in degrees is not
separable in this form. Evaluate on a direction-cosine grid fine enough for the
application, and convert the coordinates with `azel`.

## 5. Linear forms

aᴴ(u) y = Σ_n conj(g_n) y_n exp(−j κ p_n·u) is the same kind of series, over the element
positions instead of the lags: the weighted entries of y are accumulated into the
bounding box of the positions (repeated positions add), then transformed along each
axis with the opposite sign. `Lattice.linear` evaluates it for a batch of vectors, e.g.
the snapshots of a frame.

## 6. Covariance from a spectrum, and IAA

Iterative methods such as IAA also need the reverse operation: from a power p_k on the
grid, the model covariance

    R_m[m, n] = Σ_k p_k a_m(u_k) conj(a_n(u_k)) = g_m conj(g_n) Σ_k p_k exp(−j κ l·u_k),   l = p_n − p_m

Again it only depends on the lag l. So the sum over the K grid points is the adjoint of
the evaluation in §4, computed once for the Λ lags, and the N² entries are read from it.
Cost: O(K Λ) (matrix path) or O(K log K) (FFT path), instead of O(K N²).
(`Lattice.covariance_from_spectrum`.)

IAA (Yardibi et al., 2010) alternates the model covariance and an amplitude estimate per
grid point. With several snapshots y(t) and the sample covariance R̂, the power update is

    p_k = (1/T) Σ_t |a_kᴴ R_m⁻¹ y(t)|² / (a_kᴴ R_m⁻¹ a_k)² = a_kᴴ (R_m⁻¹ R̂ R_m⁻¹) a_k / (a_kᴴ R_m⁻¹ a_k)²

two quadratic forms. Every step of an iteration is therefore one of the engine's
operations: the model covariance (adjoint at the lags), one N × N inverse per matrix,
and two evaluations on the grid. The grid is IAA's dictionary of candidate directions:
it must cover every direction a source can come from (`Grid.periodic` covers all of
them), finely enough.

## 7. Cost summary

For B matrices (batch), K = K_1 … K_D grid points:

| Step | Direct | lagspec |
|---|---|---|
| per-matrix preprocessing (inverse, eigendecomposition) | O(B N³) | O(B N³) (unchanged) |
| lag sums | – | O(B N² Λ) as one product, Λ = number of lags |
| grid evaluation | O(B K N²) | O(B K Λ_D) (matrix path), O(B K log K) (FFT path), or three FFTs of length L ≥ Λ + K − 1 per axis line (chirp-z) |
| model covariance (IAA) | O(B K N²) | O(B K Λ_D) or O(B K log K), then N² reads |

The spectrum step usually dominates the direct computation. Here it shrinks by about
N² / Λ_D. The inverse or eigendecomposition is untouched; for small arrays it is cheap
and batched.

## 8. Scope and limits

- **Lattice geometry required.** For positions that are not integers, `quad_direct` gives
  the same API by explicit steering vectors.
- **Errors that are the same for every element at a given angle** (a common element
  pattern) only scale the spectrum per grid point. Per-element gains are exact (§3).
  Angle-dependent errors that differ between elements are outside the model. A manifold
  expanded on a few extra spatial harmonics would keep a Fourier structure (planned).

## References

- B. R. Musicus, "Fast MLM power spectrum estimation from uniformly spaced correlations,"
  IEEE Trans. Acoust., Speech, Signal Process., 33(4), 1985.
- Z.-S. Liu, H. Li and J. Li, "Efficient implementation of Capon and APES for spectral
  estimation," IEEE Trans. Aerosp. Electron. Syst., 34(4), 1998.
- A. Jakobsson, S. L. Marple and P. Stoica, "Computationally efficient two-dimensional
  Capon spectrum analysis," IEEE Trans. Signal Process., 48(9), 2000.
- T. Ekman, A. Jakobsson and P. Stoica, "On the efficient implementation of the Capon
  spectral estimator," EUSIPCO, 2000.
- M. Xue, L. Xu and J. Li, "IAA spectral estimation: fast implementation using the
  Gohberg–Semencul factorization," IEEE Trans. Signal Process., 59(7), 2011.
- T. Yardibi, J. Li, P. Stoica, M. Xue and A. B. Baggeroer, "Source localization and
  sensing: a nonparametric iterative adaptive approach based on weighted least
  squares," IEEE Trans. Aerosp. Electron. Syst., 46(1), 2010.
- L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform algorithm," IEEE
  Trans. Audio Electroacoust., 17(2), 1969.
- G.-O. Glentis and A. Jakobsson, "Superfast approximative implementation of the IAA
  spectral estimate," IEEE Trans. Signal Process., 60(1), 2012.
- J. Zhuang, W. Li and A. Manikas, "An IDFT-based root-MUSIC for arbitrary arrays,"
  ICASSP, 2010.
