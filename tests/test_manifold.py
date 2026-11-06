import numpy as np
import pytest

from lagspec import Grid, Lattice, Manifold, capon, covariance, iaa

RNG = np.random.default_rng(3)


def random_manifold(lat, extra=1, size=0.15):
    """the nominal manifold plus random coefficients on every harmonic"""
    m = Manifold.from_lattice(lat, extra)
    G = m.G + size * (RNG.standard_normal(m.G.shape) + 1j * RNG.standard_normal(m.G.shape))
    return Manifold(m.harmonics.positions, G, lat.spacing)


def hermitian(n, batch=()):
    x = RNG.standard_normal(batch + (n, 2 * n)) + 1j * RNG.standard_normal(batch + (n, 2 * n))
    return x @ np.swapaxes(x.conj(), -1, -2) / (2 * n)


CASES = [
    (Lattice.ula(8), [Grid.periodic(64), Grid.linspace(-0.8, 0.7, 37)]),
    (Lattice.ura(4, 4), [Grid.periodic((24, 24)), Grid.linspace((-0.6, -0.5), (0.6, 0.4), (25, 19))]),
    (Lattice.mimo([0, 1], [0, 1, 2]), [Grid.periodic(16)]),
]


@pytest.mark.parametrize("k", range(len(CASES)))
def test_ideal_manifold_reproduces_the_lattice(k):
    lat0, grids = CASES[k]
    g = np.exp(1j * RNG.uniform(-0.5, 0.5, lat0.n_elements))
    lat = Lattice(lat0.positions, lat0.spacing, gains=g)
    for extra in (0, 2):
        man = Manifold.from_lattice(lat, extra)
        M = hermitian(lat.n_elements)
        y = RNG.standard_normal(lat.n_elements) + 1j * RNG.standard_normal(lat.n_elements)
        for grid in grids:
            np.testing.assert_allclose(man.quad(M, grid), lat.quad(M, grid), atol=1e-9)
            np.testing.assert_allclose(man.linear(y, grid), lat.linear(y, grid), atol=1e-9)
            p = RNG.uniform(0, 1, grid.shape)
            np.testing.assert_allclose(man.covariance_from_spectrum(p, grid), lat.covariance_from_spectrum(p, grid), atol=1e-9)


@pytest.mark.parametrize("k", range(len(CASES)))
def test_manifold_forms_match_direct(k):
    lat, grids = CASES[k]
    man = random_manifold(lat)
    M = hermitian(lat.n_elements, (2,))
    y = RNG.standard_normal((2, lat.n_elements)) + 1j * RNG.standard_normal((2, lat.n_elements))
    for grid in grids:
        ref = man.quad_direct(M, grid)
        np.testing.assert_allclose(man.quad(M, grid), ref, atol=1e-9 * np.abs(ref).max())
        ref = man.linear_direct(y, grid)
        np.testing.assert_allclose(man.linear(y, grid), ref, atol=1e-9 * np.abs(ref).max())
        p = RNG.uniform(0, 1, (2,) + grid.shape)
        ref = man.covariance_from_spectrum_direct(p, grid)
        np.testing.assert_allclose(man.covariance_from_spectrum(p, grid), ref, atol=1e-9 * np.abs(ref).max())


def test_fit_recovers_a_manifold_from_a_table():
    lat = Lattice.ura(4, 4)
    true = random_manifold(lat, extra=1, size=0.05)
    pts = RNG.uniform(-0.6, 0.6, (300, 2))
    alpha = RNG.uniform(0.5, 2, 300) * np.exp(1j * RNG.uniform(-np.pi, np.pi, 300))  # unknown per direction
    table = alpha[:, None] * true.steering(pts)
    fit = Manifold.fit(table[:200], pts[:200], lat, extra=1, ridge=1e-9)
    assert np.max(fit.off_model(table[200:], pts[200:])) < 1e-8  # held-out directions
    als = Manifold.fit(table[:200], pts[:200], lat, extra=1, ridge=1e-6, n_iter=1000, exact=False)
    assert np.max(als.off_model(table[200:], pts[200:])) < 1e-3  # the slow fallback gets there too
    nominal = Manifold.from_lattice(lat)
    assert np.median(nominal.off_model(table[200:], pts[200:])) > 1e-3  # the ideal model does not fit


def test_capon_and_iaa_on_a_manifold_match_direct():
    lat = Lattice.ura(4, 4)
    man = random_manifold(lat, size=0.1)
    grid = Grid.linspace((-0.8, -0.8), (0.8, 0.8), (17, 15))
    R = covariance(RNG.standard_normal((2, 16, 40)) + 1j * RNG.standard_normal((2, 16, 40)))
    A = man.steering(grid.points())
    q = lambda M: np.real(np.einsum("pi,...ij,pj->...p", A.conj(), M, A)).reshape((2,) + grid.shape)  # noqa: E731
    np.testing.assert_allclose(capon(R, man, grid), 1 / q(np.linalg.inv(R)), rtol=1e-8)
    from test_methods import iaa_direct

    np.testing.assert_allclose(iaa(R, man, grid, n_iter=4), iaa_direct(R, man, grid, 4, 1e-9), rtol=1e-6, atol=1e-12)
