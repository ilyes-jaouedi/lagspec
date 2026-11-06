"""
Covariance-based spatial spectra, each one a quadratic form a^H M a evaluated by the
lag-domain engine:

    Bartlett   M = R                      P(u) = a^H R a
    Capon      M = (R + delta I)^-1       P(u) = 1 / a^H (R + delta I)^-1 a
    MUSIC      M = E_n E_n^H              P(u) = 1 / a^H E_n E_n^H a
    IAA        iterated: the model covariance sum_k p_k a_k a_k^H (an adjoint transform
               at the lags) and two quadratic forms per iteration

All functions accept a batch of covariances (..., N, N), e.g. one per range bin.
"""

import numpy as np

TINY = 1e-30


def covariance(x):
    """(..., N, S) snapshots -> (..., N, N) sample covariance"""
    x = np.asarray(x)
    return x @ np.swapaxes(x.conj(), -1, -2) / x.shape[-1]


def bartlett(R, lattice, grid, method="auto"):
    """conventional (delay-and-sum) beamformer power a^H R a"""
    return lattice.quad(R, grid, method)


def capon(R, lattice, grid, loading=0.0, method="auto"):
    """
    Capon / MVDR power 1 / a^H (R + delta I)^-1 a, with diagonal loading
    delta = loading * trace(R) / N
    """
    R = np.asarray(R)
    n = R.shape[-1]
    delta = loading * np.real(np.trace(R, axis1=-2, axis2=-1)) / n
    Ri = np.linalg.inv(R + delta[..., None, None] * np.eye(n))
    return 1.0 / np.maximum(lattice.quad(Ri, grid, method), TINY)


def music(R, lattice, grid, n_sources, method="auto"):
    """MUSIC pseudo-spectrum 1 / a^H E_n E_n^H a, E_n the N - n_sources noise eigenvectors"""
    R = np.asarray(R)
    n = R.shape[-1]
    if not 0 <= n_sources < n:
        raise ValueError(f"n_sources must be in [0, {n - 1}]")
    _, V = np.linalg.eigh(R)  # ascending eigenvalues
    En = V[..., : n - n_sources]
    P = En @ np.swapaxes(En.conj(), -1, -2)
    return 1.0 / np.maximum(lattice.quad(P, grid, method), TINY)


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
    R = np.asarray(R)
    n = R.shape[-1]
    norm = n if lattice.gains is None else float(np.sum(np.abs(lattice.gains) ** 2))
    mask = 1.0 if support is None else np.asarray(support, dtype=float)
    p = np.maximum(lattice.quad(R, grid, method), 0.0) / norm**2 * mask
    eye = np.eye(n)
    for _ in range(n_iter):
        Rm = lattice.covariance_from_spectrum(p, grid, method)
        delta = loading * np.real(np.trace(Rm, axis1=-2, axis2=-1)) / n
        Ri = np.linalg.inv(Rm + delta[..., None, None] * eye)
        num = lattice.quad(Ri @ R @ Ri, grid, method)
        den = lattice.quad(Ri, grid, method)
        p_new = np.maximum(num, 0.0) / np.maximum(den, TINY) ** 2 * mask
        done = tol is not None and np.max(np.abs(p_new - p)) <= tol * np.max(np.abs(p))
        p = p_new
        if done:
            break
    return p
