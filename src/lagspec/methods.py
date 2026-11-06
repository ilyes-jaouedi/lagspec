"""
Covariance-based spatial spectra, each one a quadratic form a^H M a evaluated by the
lag-domain engine:

    Bartlett   M = R                      P(u) = a^H R a
    Capon      M = (R + delta I)^-1       P(u) = 1 / a^H (R + delta I)^-1 a
    MUSIC      M = E_n E_n^H              P(u) = 1 / a^H E_n E_n^H a
    IAA        iterated: the model covariance sum_k p_k a_k a_k^H (an adjoint transform
               at the lags) and two quadratic forms per iteration

All functions accept a batch of covariances (..., N, N), e.g. one per range bin, and a
Lattice or a calibrated Manifold.
"""

import numpy as np

from . import _backend as B

TINY = 1e-30


def _prepare(x):
    xp = B.namespace(x)
    return xp, (np.asarray(x) if xp is np else x)


def _herm(xp, x):
    """conjugate transpose of the last two axes"""
    return np.swapaxes(x.conj(), -1, -2) if xp is np else xp.matrix_transpose(xp.conj(x))


def _eye(xp, n, like):
    return np.eye(n, dtype=like.dtype) if xp is np else xp.eye(n, dtype=like.dtype, device=B.device(like))


def _trace(xp, R):
    """real part of the trace of the last two axes"""
    return np.real(np.trace(R, axis1=-2, axis2=-1)) if xp is np else xp.real(xp.linalg.trace(R))


def _floor(xp, x, v):
    return np.maximum(x, v) if xp is np else xp.maximum(x, xp.asarray(v, dtype=x.dtype, device=B.device(x)))


def covariance(x):
    """(..., N, S) snapshots -> (..., N, N) sample covariance"""
    xp, x = _prepare(x)
    return x @ _herm(xp, x) / x.shape[-1]


def bartlett(R, lattice, grid, method="auto"):
    """conventional (delay-and-sum) beamformer power a^H R a"""
    return lattice.quad(R, grid, method)


def capon(R, lattice, grid, loading=0.0, method="auto"):
    """
    Capon / MVDR power 1 / a^H (R + delta I)^-1 a, with diagonal loading
    delta = loading * trace(R) / N
    """
    xp, R = _prepare(R)
    n = R.shape[-1]
    delta = loading * _trace(xp, R) / n
    Ri = xp.linalg.inv(R + delta[..., None, None] * _eye(xp, n, R))
    return 1.0 / _floor(xp, lattice.quad(Ri, grid, method), TINY)


def music(R, lattice, grid, n_sources, method="auto"):
    """MUSIC pseudo-spectrum 1 / a^H E_n E_n^H a, E_n the N - n_sources noise eigenvectors"""
    xp, R = _prepare(R)
    n = R.shape[-1]
    if not 0 <= n_sources < n:
        raise ValueError(f"n_sources must be in [0, {n - 1}]")
    V = xp.linalg.eigh(R)[1]  # ascending eigenvalues
    En = V[..., : n - n_sources]
    return 1.0 / _floor(xp, lattice.quad(En @ _herm(xp, En), grid, method), TINY)


def iaa(R, lattice, grid, n_iter=15, loading=1e-9, support=None, tol=None, method="auto"):
    """
    Iterative adaptive approach (Yardibi, Li, Stoica, Xue and Baggeroer, 2010), the
    multi-snapshot form written with the sample covariance R:

        R_m = sum_k p_k a_k a_k^H                       (model covariance on the grid)
        p_k = a_k^H R_m^-1 R R_m^-1 a_k / (a_k^H R_m^-1 a_k)^2

    started from the normalised Bartlett spectrum a_k^H R a_k / |a_k|^4. Works with few
    snapshots (even one), where Capon's sample covariance is singular.

    The grid is the dictionary of candidate directions: it must cover every direction a
    source can come from (Grid.periodic covers all of them), finely enough. support
    (grid-shaped, boolean) restricts the dictionary, e.g. to Grid.visible().
    loading   relative diagonal loading of R_m, for numerical stability only
    tol       stop when the largest relative change of p falls below tol
    """
    xp, R = _prepare(R)
    n = R.shape[-1]
    eye = _eye(xp, n, R)
    norm = lattice.quad(eye, grid, method)  # |a(u)|^2 (varies with u on a Manifold)
    mask = 1.0 if support is None else B.asarray(xp, np.asarray(support, dtype=float), norm, norm.dtype)
    p = _floor(xp, lattice.quad(R, grid, method), 0.0) / norm**2 * mask
    for _ in range(n_iter):
        Rm = lattice.covariance_from_spectrum(p, grid, method)
        delta = loading * _trace(xp, Rm) / n
        Ri = xp.linalg.inv(Rm + delta[..., None, None] * _eye(xp, n, Rm))
        num = lattice.quad(Ri @ R @ Ri, grid, method)
        den = lattice.quad(Ri, grid, method)
        p_new = _floor(xp, num, 0.0) / _floor(xp, den, TINY) ** 2 * mask
        done = tol is not None and bool(xp.max(xp.abs(p_new - p)) <= tol * xp.max(xp.abs(p)))
        p = p_new
        if done:
            break
    return p
