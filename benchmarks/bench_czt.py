"""
Crossover between the matrix-product and chirp-z paths of one axis (non-periodic grid),
used to set transform.CZT_FACTOR: chirp-z wins when n_offsets * K > CZT_FACTOR * L log2 L.

    uv run python benchmarks/bench_czt.py
"""

import time

import numpy as np

from lagspec import transform as T

BATCH = 64


def best_of(f, repeat=5):
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    rng = np.random.default_rng(0)
    print(f"{'offsets':>8s} {'grid':>6s} {'matmul':>10s} {'czt':>10s} {'czt/matmul':>11s} {'I K / L log2 L':>15s}")
    for n_off in (8, 16, 32, 64, 128, 256, 1024):
        for num in (64, 256, 1024, 4096):
            offsets = np.arange(n_off) - n_off // 2
            axis = (-0.3, 0.6 / num, num)  # a zoom, not periodic
            c = rng.standard_normal((BATCH, n_off)) + 1j * rng.standard_normal((BATCH, n_off))
            t_m = best_of(lambda: T.forward(c, offsets, axis, np.pi, 1, "matmul"))
            t_c = best_of(lambda: T.forward(c, offsets, axis, np.pi, 1, "czt"))
            L = T._fft_length(n_off + num - 1)
            print(f"{n_off:8d} {num:6d} {t_m * 1e3:8.2f}ms {t_c * 1e3:8.2f}ms {t_c / t_m:11.2f} {n_off * num / (L * np.log2(L)):15.2f}")


if __name__ == "__main__":
    main()
