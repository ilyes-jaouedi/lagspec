"""
Covariance-based spatial spectra, each one a quadratic form a^H M a evaluated by the
lag-domain engine:

    Bartlett   M = R                      P(u) = a^H R a
    Capon      M = (R + delta I)^-1       P(u) = 1 / a^H (R + delta I)^-1 a
    MUSIC      M = E_n E_n^H              P(u) = 1 / a^H E_n E_n^H a

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
