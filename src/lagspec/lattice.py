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

from . import transform as T
from .grid import Grid


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
        self._lag_index = np.ravel_multi_index(tuple((diff - self.lag_min).T), self.lag_shape)
        self._lag_sum = T.Accumulator(self._lag_index, np.prod(self.lag_shape))

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
        lags = self._lag_sum(M.reshape(M.shape[:-2] + (n * n,)))
        return lags.reshape(M.shape[:-2] + self.lag_shape)

    def evaluate(self, lags, grid, method="auto"):
        """
        sum_l L(l) exp(j kappa l . u) on the grid: (..., *lag_shape) -> (..., *grid.shape)

        method  "matmul": one small matrix product per axis (any uniform grid)
                "fft": scatter the lags onto the grid and inverse FFT (needs a periodic
                       grid, see Grid.periodic)
                "auto": FFT on the periodic axes where it is cheaper
        """
        return self._series(lags, self.lag_min, self.lag_shape, grid, +1, method, T.forward)

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

    def _series(self, arr, offset_min, shape, grid, sign, method, transform):
        """
        apply `transform` (forward: offsets -> grid, adjoint: grid -> offsets) along each
        axis of the last ndim axes of arr
        """
        self._check_grid(grid)
        arr = np.asarray(arr, dtype=complex)
        nb = arr.ndim - self.ndim
        expected = shape if transform is T.forward else grid.shape
        if nb < 0 or arr.shape[nb:] != tuple(expected):
            raise ValueError(f"expected (..., {tuple(expected)}), got {arr.shape}")
        for d in range(self.ndim):
            offsets = offset_min[d] + np.arange(shape[d])
            arr = np.moveaxis(arr, nb + d, -1)
            arr = transform(arr, offsets, grid.params[d], self.kappa, sign, method)
            arr = np.moveaxis(arr, -1, nb + d)
        return arr

    def __repr__(self):
        return (
            f"Lattice({self.n_elements} elements, {self.ndim}-D, spacing {self.spacing:g}, "
            f"lags {self.lag_shape})"
        )
