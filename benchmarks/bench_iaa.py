"""
Run time of IAA: lagspec against explicit steering vectors (same iterations, same result).

    uv run python benchmarks/bench_iaa.py
"""

import time

import numpy as np

from lagspec import Grid, Lattice, covariance, iaa

N_ITER = 10
CASES = [
    ("URA 4x4, 32 matrices, 61x61 grid", Lattice.ura(4, 4), 32, Grid.linspace((-0.9, -0.9), (0.9, 0.9), (61, 61))),
    ("ULA 16, 128 matrices, 512 periodic", Lattice.ula(16), 128, Grid.periodic(512)),
]


def iaa_direct(R, lat, grid, n_iter, loading=1e-9):
    A = lat.steering(grid.points())
    n = A.shape[1]
    q = lambda M: np.real(np.einsum("pi,...ij,pj->...p", A.conj(), M, A, optimize=True))  # noqa: E731
    p = np.maximum(q(R), 0) / n**2
    for _ in range(n_iter):
        Rm = np.einsum("...p,pi,pj->...ij", p, A, A.conj(), optimize=True)
        delta = loading * np.real(np.trace(Rm, axis1=-2, axis2=-1)) / n
        Ri = np.linalg.inv(Rm + delta[..., None, None] * np.eye(n))
        p = np.maximum(q(Ri @ R @ Ri), 0) / q(Ri) ** 2
    return p.reshape(R.shape[:-2] + grid.shape)


def main():
    rng = np.random.default_rng(0)
    print(f"{'case':40s} {'direct':>9s} {'lagspec':>9s} {'speed-up':>9s} {'max rel. diff':>14s}")
    for name, lat, b, grid in CASES:
        n = lat.n_elements
        R = covariance(rng.standard_normal((b, n, 8)) + 1j * rng.standard_normal((b, n, 8)))
        t0 = time.perf_counter()
        ref = iaa_direct(R, lat, grid, N_ITER)
        t_direct = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = iaa(R, lat, grid, n_iter=N_ITER)
        t_fast = time.perf_counter() - t0
        diff = np.max(np.abs(p - ref) / np.maximum(np.abs(ref), 1e-12 * np.abs(ref).max()))
        print(f"{name:40s} {t_direct:8.2f}s {t_fast:8.3f}s {t_direct / t_fast:8.0f}x {diff:14.1e}")


if __name__ == "__main__":
    main()
