"""
Run time on a CUDA GPU (PyTorch) against NumPy on the CPU, for a radar-like batch: one
covariance per range bin, Capon and IAA on a dense direction grid.

    python benchmarks/bench_gpu.py
"""

import time

import numpy as np

from lagspec import Grid, Lattice, capon, covariance, iaa

CASES = [
    ("Capon, 4x4, 128 bins, 61x61", Lattice.ura(4, 4), 128, Grid.linspace((-0.6, -0.6), (0.6, 0.6), (61, 61)), "capon"),
    ("Capon, 4x4, 128 bins, 256x256", Lattice.ura(4, 4), 128, Grid.periodic((256, 256)), "capon"),
    ("Capon, 8x8, 128 bins, 256x256", Lattice.ura(8, 8), 128, Grid.periodic((256, 256)), "capon"),
    ("IAA 10 it., 4x4, 128 bins, 61x61", Lattice.ura(4, 4), 128, Grid.linspace((-0.9, -0.9), (0.9, 0.9), (61, 61)), "iaa"),
]


def run(kind, R, lat, grid):
    if kind == "capon":
        return capon(R, lat, grid, loading=1e-3)
    return iaa(R, lat, grid, n_iter=10)


def timed(f, sync, repeat=5):
    f()  # warm-up (CUDA context, cuFFT plans, constants on the device)
    sync()
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        sync()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    rng = np.random.default_rng(0)
    print(f"{'case':36s} {'numpy cpu':>10s} {'torch c128':>11s} {'torch c64':>10s} {'speed-up c64':>13s}")
    for name, lat, b, grid, kind in CASES:
        n = lat.n_elements
        x = rng.standard_normal((b, n, 40)) + 1j * rng.standard_normal((b, n, 40))
        R = covariance(x)
        t_np = timed(lambda: run(kind, R, lat, grid), lambda: None, repeat=2)
        R128 = torch.from_numpy(R).to("cuda")
        R64 = R128.to(torch.complex64)
        t_128 = timed(lambda: run(kind, R128, lat, grid), torch.cuda.synchronize)
        t_64 = timed(lambda: run(kind, R64, lat, grid), torch.cuda.synchronize)
        print(f"{name:36s} {t_np * 1e3:8.1f}ms {t_128 * 1e3:9.1f}ms {t_64 * 1e3:8.1f}ms {t_np / t_64:12.0f}x")


if __name__ == "__main__":
    main()
