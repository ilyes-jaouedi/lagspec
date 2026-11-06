"""
Arrays whose elements sit on an integer lattice, and the lag-domain engine that
evaluates a(u)^H M a(u) on a whole grid at once.

Element n sits at the integer position p_n (in units of the lattice step) and has the
steering phase exp(j kappa p_n . u), kappa = 2 pi spacing / wavelength. Then

    a(u)^H M a(u) = sum_{m, n} conj(g_m) g_n M[m, n] exp(j kappa (p_n - p_m) . u)
                  = sum_l L(l) exp(j kappa l . u)

where L(l) sums the (gain-weighted) entries of M whose position difference is the lag
l: a Fourier series over the difference co-array. On a grid uniform in each
direction-cosine axis it is evaluated exactly, one axis at a time, either with a small
matrix product or with an FFT. See docs/theory.md.
"""

import numpy as np

from .grid import Grid

# method="auto" uses the FFT on a periodic axis when its number of lags exceeds
# FFT_LAG_FACTOR * log2(grid size): below that, the small BLAS product is faster
# (benchmarks/bench_quad.py)
FFT_LAG_FACTOR = 8.0
# largest summation matrix (N^2 x number of lags entries) kept dense
DENSE_SUM_LIMIT = 4_000_000


class Lattice:
    """
    positions  (N,) or (N, D) element positions in units of the lattice step; must be
               integers (within atol). Repeated positions (MIMO virtual arrays) are
               allowed.
    spacing    lattice step in wavelengths (0.5: half-wavelength lattice)
    gains      optional (N,) complex per-element gains of the steering vectors
               (calibration); folded into M, so the evaluation stays exact
    """

    def __init__(self, positions, spacing=0.5, gains=None, atol=1e-6):
        p = np.asarray(positions, dtype=float)
        if p.ndim == 1:
            p = p[:, None]
        if p.ndim != 2 or len(p) == 0:
            raise ValueError("positions must have shape (N,) or (N, D)")
        r = np.rint(p)
        if np.max(np.abs(p - r)) > atol:
            raise ValueError(
                "element positions are not on the integer lattice: express them in units "
                "of the lattice step, or use quad_direct"
            )
        self.positions = r.astype(np.int64)
        self.spacing = float(spacing)
        self.gains = None if gains is None else np.asarray(gains, dtype=complex).reshape(-1)
        if self.gains is not None and len(self.gains) != len(self.positions):
            raise ValueError("gains must have one entry per element")

        n, d = self.positions.shape
        diff = (self.positions[None, :, :] - self.positions[:, None, :]).reshape(-1, d)  # p_n - p_m
        self.lag_min = diff.min(axis=0)
        self.lag_shape = tuple(int(x) for x in diff.max(axis=0) - self.lag_min + 1)
        flat = np.ravel_multi_index(tuple((diff - self.lag_min).T), self.lag_shape)
        self._flat = flat
        self._n_lags = int(np.prod(self.lag_shape))
        # small arrays: a (N*N, number of lags) 0/1 matrix, lag sums of a batch by one
        # BLAS product; large arrays: accumulation by index (the matrix grows as N^4)
        self._sum = None
        if n * n * self._n_lags <= DENSE_SUM_LIMIT:
            self._sum = np.zeros((n * n, self._n_lags))
            self._sum[np.arange(n * n), flat] = 1.0

    # -- constructors ---------------------------------------------------------------

    @classmethod
    def ula(cls, n, spacing=0.5, gains=None):
        """uniform linear array of n elements"""
        return cls(np.arange(n), spacing, gains)

    @classmethod
    def ura(cls, nx, ny, spacing=0.5, gains=None):
        """uniform rectangular array; element index ix * ny + iy"""
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        return cls(np.column_stack((ix.ravel(), iy.ravel())), spacing, gains)

    @classmethod
    def mimo(cls, tx, rx, spacing=0.5, gains=None):
        """
        virtual array of a MIMO radar: element t * len(rx) + r at tx[t] + rx[r]
        (positions in units of the lattice step, (T,) / (R,) or (T, D) / (R, D))
        """
        tx, rx = np.asarray(tx, float), np.asarray(rx, float)
        if tx.ndim == 1:
            tx, rx = tx[:, None], rx[:, None]
        return cls((tx[:, None, :] + rx[None, :, :]).reshape(-1, tx.shape[1]), spacing, gains)

    # -- properties -----------------------------------------------------------------

    @property
    def n_elements(self):
        return len(self.positions)

    @property
    def ndim(self):
        return self.positions.shape[1]

    @property
    def kappa(self):
        """phase per lattice step and unit direction cosine"""
        return 2 * np.pi * self.spacing

    def lags(self, d):
        """the lags along axis d covered by the lag-sum array"""
        return self.lag_min[d] + np.arange(self.lag_shape[d])

    # -- engine ---------------------------------------------------------------------

    def lag_sums(self, M):
        """(..., N, N) matrices -> (..., *lag_shape) lag sums L(l)"""
        M = np.asarray(M)
        n = self.n_elements
        if M.shape[-2:] != (n, n):
            raise ValueError(f"expected (..., {n}, {n}) matrices, got {M.shape}")
        if self.gains is not None:
            M = self.gains.conj()[:, None] * M * self.gains[None, :]
        batch = M.shape[:-2]
        flat = M.reshape(-1, n * n)
        if self._sum is not None:
            lags = flat @ self._sum
        else:
            b = len(flat)
            idx = (np.arange(b)[:, None] * self._n_lags + self._flat[None, :]).ravel()
            size = b * self._n_lags
            lags = np.bincount(idx, flat.real.ravel(), size)
            if np.iscomplexobj(flat):
                lags = lags + 1j * np.bincount(idx, flat.imag.ravel(), size)
            lags = lags.reshape(b, self._n_lags)
        return lags.reshape(batch + self.lag_shape)

    def evaluate(self, lags, grid, method="auto"):
        """
        sum_l L(l) exp(j kappa l . u) on the grid: (..., *lag_shape) -> (..., *grid.shape)

        method  "matmul": one small matrix product per axis (any uniform grid)
                "fft": scatter the lags onto the grid and inverse FFT (needs a periodic
                       grid, see Grid.periodic)
                "auto": FFT on the periodic axes where it is cheaper
        """
        self._check_grid(grid)
        arr = np.asarray(lags, dtype=complex)
        nb = arr.ndim - self.ndim
        if nb < 0 or arr.shape[nb:] != self.lag_shape:
            raise ValueError(f"expected (..., {self.lag_shape}) lag sums, got {arr.shape}")
        for d in range(self.ndim):
            start, step, num = grid.params[d]
            ell = self.lags(d)
            periodic = self._periodic(step, num)
            if method == "fft" and not periodic:
                raise ValueError(f"axis {d} is not periodic (use Grid.periodic) for method='fft'")
            use_fft = method == "fft" or (
                method == "auto" and periodic and len(ell) > FFT_LAG_FACTOR * np.log2(max(num, 2))
            )
            arr = np.moveaxis(arr, nb + d, -1)
            if use_fft:
                arr = self._fft_axis(arr, ell, start, num)
            else:
                arr = arr @ np.exp(1j * self.kappa * np.outer(ell, start + step * np.arange(num)))
            arr = np.moveaxis(arr, -1, nb + d)
        return arr

    def quad(self, M, grid, method="auto", real=True):
        """
        a(u)^H M a(u) for every direction u of the grid and every matrix of the batch:
        (..., N, N) -> (..., *grid.shape). real=True returns the real part (the value is
        real for Hermitian M).
        """
        q = self.evaluate(self.lag_sums(M), grid, method)
        return q.real if real else q

    # -- direct reference -----------------------------------------------------------

    def steering(self, points):
        """(P, D) direction cosines -> (P, N) steering vectors"""
        points = np.asarray(points, dtype=float).reshape(-1, self.ndim)
        a = np.exp(1j * self.kappa * points @ self.positions.T)
        return a if self.gains is None else a * self.gains

    def quad_direct(self, M, grid, real=True):
        """the same as quad, by explicit steering vectors (reference, any geometry)"""
        A = self.steering(grid.points())
        M = np.asarray(M)
        q = np.einsum("pi,...ij,pj->...p", A.conj(), M, A, optimize=True)
        q = q.reshape(M.shape[:-2] + grid.shape)
        return q.real if real else q

    # -- helpers --------------------------------------------------------------------

    def _check_grid(self, grid):
        if not isinstance(grid, Grid) or grid.ndim != self.ndim:
            raise ValueError(f"expected a Grid with {self.ndim} axes")

    def _periodic(self, step, num):
        return np.isclose(self.kappa * step * num, 2 * np.pi, rtol=1e-9, atol=0.0)

    def _fft_axis(self, arr, ell, start, num):
        """last axis: lags -> grid, with kappa * step * num = 2 pi"""
        phased = arr * np.exp(1j * self.kappa * ell * start)
        out = np.zeros(arr.shape[:-1] + (num,), dtype=complex)
        bins = ell % num
        if len(ell) <= num:  # consecutive lags: distinct bins
            out[..., bins] = phased
        else:
            for j, k in enumerate(bins):  # lags that alias onto one bin add up (exact)
                out[..., k] += phased[..., j]
        return np.fft.ifft(out, axis=-1) * num

    def __repr__(self):
        return (
            f"Lattice({self.n_elements} elements, {self.ndim}-D, spacing {self.spacing:g}, "
            f"lags {self.lag_shape})"
        )
