"""
The forms and methods on a CUDA GPU, through PyTorch and CuPy. Skipped without a GPU.
Results must match NumPy and stay on the GPU, in the input's precision.
"""

import numpy as np
import pytest

from lagspec import Grid, Lattice, Manifold, capon, covariance, iaa, music

pytest.importorskip("array_api_compat")

RNG = np.random.default_rng(5)


def gpu_backends():
    out = []
    try:
        import torch

        if torch.cuda.is_available():
            out.append(pytest.param("torch", id="torch-cuda"))
    except ImportError:
        pass
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            out.append(pytest.param("cupy", id="cupy"))
    except Exception:  # noqa: BLE001 (no CUDA driver, no device, ...)
        pass
    return out


BACKENDS = gpu_backends()
if not BACKENDS:
    pytest.skip("no CUDA GPU backend available", allow_module_level=True)


def to_gpu(backend, a):
    a = np.ascontiguousarray(a)
    if backend == "torch":
        import torch

        return torch.from_numpy(a).to("cuda")
    import cupy

    return cupy.asarray(a)


def on_gpu(backend, x):
    if backend == "torch":
        return x.is_cuda
    import cupy

    return isinstance(x, cupy.ndarray)


def to_np(backend, x):
    return x.cpu().numpy() if backend == "torch" else x.get()


def check(backend, x, ref, rtol=1e-9):
    assert on_gpu(backend, x)
    np.testing.assert_allclose(to_np(backend, x), ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def snapshots(n, batch, snap=40):
    return RNG.standard_normal(batch + (n, snap)) + 1j * RNG.standard_normal(batch + (n, snap))


GRIDS = {"matmul": Grid.linspace((-0.7, -0.6), (0.7, 0.6), (21, 19)), "fft": Grid.periodic((32, 32))}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", ["matmul", "fft", "czt"])
def test_engine_forms(backend, path):
    lat = Lattice.ura(4, 4, gains=np.exp(1j * np.linspace(-0.3, 0.3, 16)))
    grid = GRIDS["fft" if path == "fft" else "matmul"]
    R = covariance(snapshots(16, (3,)))
    y = snapshots(16, (2,), 1)[..., 0]
    p = RNG.uniform(0, 1, (2,) + grid.shape)
    check(backend, lat.quad(to_gpu(backend, R), grid, path), lat.quad(R, grid, path))
    check(backend, lat.linear(to_gpu(backend, y), grid, path), lat.linear(y, grid, path))
    check(backend, lat.covariance_from_spectrum(to_gpu(backend, p), grid, path), lat.covariance_from_spectrum(p, grid, path))


@pytest.mark.parametrize("backend", BACKENDS)
def test_methods_and_manifold(backend):
    lat = Lattice.ura(4, 4)
    m = Manifold.from_lattice(lat, 1)
    man = Manifold(m.harmonics.positions, m.G + 0.1 * RNG.standard_normal(m.G.shape), lat.spacing)
    grid = GRIDS["matmul"]
    R = covariance(snapshots(16, (2,)))
    Rg = to_gpu(backend, R)
    check(backend, capon(Rg, lat, grid, loading=1e-3), capon(R, lat, grid, loading=1e-3), rtol=1e-8)
    check(backend, music(Rg, lat, grid, 2), music(R, lat, grid, 2), rtol=1e-6)
    check(backend, iaa(Rg, lat, grid, n_iter=4), iaa(R, lat, grid, n_iter=4), rtol=1e-6)
    check(backend, capon(Rg, man, grid), capon(R, man, grid), rtol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_precision(backend):
    lat = Lattice.ura(4, 4)
    grid = GRIDS["fft"]
    R = covariance(snapshots(16, (2,)))
    q = capon(to_gpu(backend, R.astype(np.complex64)), lat, grid)
    assert on_gpu(backend, q) and str(q.dtype).endswith("float32")
    np.testing.assert_allclose(to_np(backend, q), capon(R, lat, grid), rtol=2e-3)
