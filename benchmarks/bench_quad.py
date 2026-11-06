"""
Run time of a^H M a on a grid: lagspec engine against explicit steering vectors.

    uv run python benchmarks/bench_quad.py
"""

import time

import numpy as np

from lagspec import Grid, Lattice

CASES = [
    ("URA 4x4, 128 matrices, 61x61 grid", Lattice.ura(4, 4), 128, Grid.linspace((-0.6, -0.6), (0.6, 0.6), (61, 61))),
    ("URA 4x4, 128 matrices, 256x256 periodic", Lattice.ura(4, 4), 128, Grid.periodic((256, 256))),
    ("URA 8x8, 16 matrices, 128x128 periodic", Lattice.ura(8, 8), 16, Grid.periodic((128, 128))),
    ("ULA 32, 512 matrices, 1024 periodic", Lattice.ula(32), 512, Grid.periodic(1024)),
    ("ULA 256, 8 matrices, 4096 periodic", Lattice.ula(256), 8, Grid.periodic(4096)),
]


def best_of(f, repeat=3):
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    rng = np.random.default_rng(0)
    print(f"{'case':42s} {'direct':>10s} {'matmul':>10s} {'auto':>10s} {'speed-up':>9s}")
    for name, lat, b, grid in CASES:
        n = lat.n_elements
        x = rng.standard_normal((b, n, 2 * n)) + 1j * rng.standard_normal((b, n, 2 * n))
        M = x @ np.swapaxes(x.conj(), -1, -2)
        t_direct = best_of(lambda: lat.quad_direct(M, grid), repeat=1)
        t_mat = best_of(lambda: lat.quad(M, grid, "matmul"))
        t_auto = best_of(lambda: lat.quad(M, grid, "auto"))
        print(f"{name:42s} {t_direct:9.3f}s {t_mat:9.4f}s {t_auto:9.4f}s {t_direct / min(t_mat, t_auto):8.0f}x")


if __name__ == "__main__":
    main()
