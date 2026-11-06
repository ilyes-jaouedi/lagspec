import numpy as np
import pytest

from lagspec import Grid, Lattice

RNG = np.random.default_rng(0)


def hermitian(n, batch=()):
    x = RNG.standard_normal(batch + (n, 2 * n)) + 1j * RNG.standard_normal(batch + (n, 2 * n))
    return x @ np.swapaxes(x.conj(), -1, -2) / (2 * n)


def general(n, batch=()):
    return RNG.standard_normal(batch + (n, n)) + 1j * RNG.standard_normal(batch + (n, n))


LATTICES = {
    "ula8": (Lattice.ula(8), [Grid.periodic(64), Grid.linspace(-0.8, 0.7, 37)]),
    "ura4x4": (Lattice.ura(4, 4), [Grid.periodic((32, 48)), Grid.linspace((-0.6, -0.5), (0.6, 0.4), (25, 19))]),
    "mimo_repeated": (Lattice.mimo([0, 1], [0, 1, 2]), [Grid.periodic(16), Grid.linspace(-1, 1, 41)]),
    "sparse_nested": (Lattice([0, 1, 2, 3, 7, 11]), [Grid.periodic(128), Grid.linspace(-0.9, 0.9, 50)]),
    "wide_spacing": (Lattice.ula(5, spacing=1.0), [Grid.periodic(40, spacing=1.0), Grid.linspace(-0.5, 0.5, 21)]),
    "3d": (
        Lattice([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 1, 1], [2, 0, 1]]),
        [Grid.periodic((8, 6, 10)), Grid.linspace((-0.5, -0.4, -0.3), (0.5, 0.4, 0.3), (7, 5, 9))],
    ),
}


@pytest.mark.parametrize("name", LATTICES)
def test_quad_matches_direct(name):
    lat, grids = LATTICES[name]
    M = hermitian(lat.n_elements)
    for grid in grids:
        ref = lat.quad_direct(M, grid)
        for method in ("auto", "matmul"):
            np.testing.assert_allclose(lat.quad(M, grid, method), ref, atol=1e-9 * np.abs(ref).max())


@pytest.mark.parametrize("name", LATTICES)
def test_fft_path_on_periodic_grid(name):
    lat, grids = LATTICES[name]
    M = general(lat.n_elements)  # non-Hermitian: complex output, both paths
    grid = grids[0]
    ref = lat.quad_direct(M, grid, real=False)
    np.testing.assert_allclose(lat.quad(M, grid, "fft", real=False), ref, atol=1e-9 * np.abs(ref).max())


def test_fft_needs_periodic_grid():
    lat = Lattice.ula(4)
    with pytest.raises(ValueError):
        lat.quad(hermitian(4), Grid.linspace(-0.5, 0.5, 11), method="fft")


def test_batch_shapes():
    lat = Lattice.ura(4, 4)
    grid = Grid.linspace((-0.5, -0.5), (0.5, 0.5), (9, 11))
    M = hermitian(16, (3, 2))
    q = lat.quad(M, grid)
    assert q.shape == (3, 2, 9, 11)
    np.testing.assert_allclose(q[1, 0], lat.quad(M[1, 0], grid), atol=1e-10)


def test_gains_are_exact():
    g = np.exp(1j * RNG.uniform(-0.5, 0.5, 16)) * RNG.uniform(0.8, 1.2, 16)
    lat = Lattice.ura(4, 4, gains=g)
    grid = Grid.periodic((16, 16))
    M = hermitian(16)
    ref = lat.quad_direct(M, grid)
    np.testing.assert_allclose(lat.quad(M, grid), ref, atol=1e-9 * np.abs(ref).max())


def test_data_side_calibration_equivalence():
    """Capon with gains in the steering == Capon with ideal steering on whitened data"""
    from lagspec import capon

    g = np.exp(1j * RNG.uniform(-0.5, 0.5, 8)) * RNG.uniform(0.8, 1.2, 8)
    grid = Grid.linspace(-0.9, 0.9, 61)
    R = hermitian(8)
    calibrated = capon(R, Lattice.ula(8, gains=g), grid)
    D = np.diag(1 / g)
    whitened = capon(D @ R @ D.conj().T, Lattice.ula(8), grid)
    np.testing.assert_allclose(calibrated, whitened, rtol=1e-9)


def test_off_lattice_positions_rejected():
    with pytest.raises(ValueError):
        Lattice([0, 1, 2.5])


def test_wrong_matrix_size_rejected():
    with pytest.raises(ValueError):
        Lattice.ula(4).quad(np.eye(5), Grid.periodic(8))


@pytest.mark.parametrize("name", ["ura4x4", "mimo_repeated", "3d"])
def test_index_accumulation_path(monkeypatch, name):
    """the large-array lag sums (no dense summation matrix) give the same result"""
    import lagspec.lattice as L

    monkeypatch.setattr(L, "DENSE_SUM_LIMIT", 0)
    lat0, grids = LATTICES[name]
    lat = Lattice(lat0.positions, lat0.spacing)
    assert lat._sum is None
    M = general(lat.n_elements, (3,))
    np.testing.assert_allclose(lat.lag_sums(M), lat0.lag_sums(M), atol=1e-12)
    np.testing.assert_allclose(lat.lag_sums(M.real), lat0.lag_sums(M.real), atol=1e-12)
