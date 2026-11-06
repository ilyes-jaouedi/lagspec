"""
Every form and method on non-NumPy arrays (array API): array-api-strict (the reference
implementation, which rejects anything outside the standard) and PyTorch when installed.
Results must match NumPy and keep the input's namespace and precision.
"""

import numpy as np
import pytest

from lagspec import Grid, Lattice, Manifold, bartlett, capon, covariance, iaa, music

pytest.importorskip("array_api_compat")

RNG = np.random.default_rng(4)


def backends():
    out = []
    try:
        import array_api_strict

        out.append(pytest.param(array_api_strict, id="array_api_strict"))
    except ImportError:
        pass
    try:
        import torch

        out.append(pytest.param(torch, id="torch"))
    except ImportError:
        pass
    return out


BACKENDS = backends()
if not BACKENDS:
    pytest.skip("no array-API backend installed", allow_module_level=True)


def to_xp(xp, a):
    a = np.ascontiguousarray(a)
    return xp.asarray(a) if xp.__name__ != "torch" else xp.from_numpy(a)


def to_np(x):
    if hasattr(x, "numpy") and not hasattr(x, "__array_namespace__"):
        return x.numpy()
    try:
        return np.from_dlpack(x)
    except Exception:  # noqa: BLE001
        return x.numpy()


def check(x, ref, xp, rtol=1e-9):
    assert type(x).__module__.split(".")[0] == xp.__name__.split(".")[0]
    np.testing.assert_allclose(to_np(x), ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def snapshots(n, batch, snap=40):
    return RNG.standard_normal(batch + (n, snap)) + 1j * RNG.standard_normal(batch + (n, snap))


GRIDS = {
    "matmul": Grid.linspace((-0.7, -0.6), (0.7, 0.6), (21, 19)),
    "fft": Grid.periodic((32, 32)),
}


@pytest.mark.parametrize("xp", BACKENDS)
@pytest.mark.parametrize("path", ["matmul", "fft", "czt"])
def test_engine_forms(xp, path):
    lat = Lattice.ura(4, 4, gains=np.exp(1j * np.linspace(-0.3, 0.3, 16)))
    grid = GRIDS["fft" if path == "fft" else "matmul"]
    R = covariance(snapshots(16, (3,)))
    y = snapshots(16, (2,), 1)[..., 0]
    p = RNG.uniform(0, 1, (2,) + grid.shape)
    check(lat.quad(to_xp(xp, R), grid, path), lat.quad(R, grid, path), xp)
    check(lat.linear(to_xp(xp, y), grid, path), lat.linear(y, grid, path), xp)
    check(lat.covariance_from_spectrum(to_xp(xp, p), grid, path), lat.covariance_from_spectrum(p, grid, path), xp)


@pytest.mark.parametrize("xp", BACKENDS)
def test_aliasing_fft_path(xp):
    lat = Lattice.ula(8)  # 15 lags on an 8-point periodic grid
    grid = Grid.periodic(8)
    R = covariance(snapshots(8, ()))
    check(lat.quad(to_xp(xp, R), grid, "fft"), lat.quad(R, grid, "fft"), xp)


@pytest.mark.parametrize("xp", BACKENDS)
def test_methods(xp):
    lat = Lattice.ura(4, 4)
    grid = GRIDS["matmul"]
    R = covariance(snapshots(16, (2,)))
    Rx = to_xp(xp, R)
    check(bartlett(Rx, lat, grid), bartlett(R, lat, grid), xp)
    check(capon(Rx, lat, grid, loading=1e-3), capon(R, lat, grid, loading=1e-3), xp, rtol=1e-8)
    check(music(Rx, lat, grid, 2), music(R, lat, grid, 2), xp, rtol=1e-6)
    check(iaa(Rx, lat, grid, n_iter=4), iaa(R, lat, grid, n_iter=4), xp, rtol=1e-6)
    x = snapshots(16, ())
    check(covariance(to_xp(xp, x)), covariance(x), xp)


@pytest.mark.parametrize("xp", BACKENDS)
def test_manifold(xp):
    lat = Lattice.ura(4, 4)
    m = Manifold.from_lattice(lat, 1)
    man = Manifold(m.harmonics.positions, m.G + 0.1 * RNG.standard_normal(m.G.shape), lat.spacing)
    grid = GRIDS["matmul"]
    R = covariance(snapshots(16, (2,)))
    check(capon(to_xp(xp, R), man, grid), capon(R, man, grid), xp, rtol=1e-8)
    check(iaa(to_xp(xp, R), man, grid, n_iter=3), iaa(R, man, grid, n_iter=3), xp, rtol=1e-6)


@pytest.mark.parametrize("xp", BACKENDS)
def test_single_precision_stays_single(xp):
    lat = Lattice.ura(4, 4)
    grid = GRIDS["fft"]
    R = covariance(snapshots(16, (2,)))
    R32 = to_xp(xp, R.astype(np.complex64))
    q = capon(R32, lat, grid)
    assert q.dtype == xp.float32
    np.testing.assert_allclose(to_np(q), capon(R, lat, grid), rtol=2e-3)
