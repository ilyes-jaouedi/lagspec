import numpy as np
import pytest

from lagspec import Grid, Lattice, bartlett, capon, covariance, music

RNG = np.random.default_rng(1)


def snapshots(lat, sources_u, n_snap=400, snr_db=20.0):
    """independent sources at the given direction cosines (1-D), white noise"""
    A = lat.steering(np.asarray(sources_u)[:, None])  # (K, N)
    s = (RNG.standard_normal((len(sources_u), n_snap)) + 1j * RNG.standard_normal((len(sources_u), n_snap))) / np.sqrt(2)
    noise = (RNG.standard_normal((lat.n_elements, n_snap)) + 1j * RNG.standard_normal((lat.n_elements, n_snap))) / np.sqrt(2)
    return A.T @ s + noise * 10 ** (-snr_db / 20)


def two_peaks(p, u):
    i = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    i = i[np.argsort(p[i])[::-1][:2]]
    return np.sort(u[i])


@pytest.mark.parametrize("method", ["matmul", "fft"])
def test_methods_match_direct_formulas(method):
    lat = Lattice.ura(4, 4)
    grid = Grid.periodic((32, 32))
    R = covariance(RNG.standard_normal((5, 16, 40)) + 1j * RNG.standard_normal((5, 16, 40)))
    A = lat.steering(grid.points())
    quad = lambda M: np.real(np.einsum("pi,...ij,pj->...p", A.conj(), M, A)).reshape((5, 32, 32))  # noqa: E731
    np.testing.assert_allclose(bartlett(R, lat, grid, method), quad(R), rtol=1e-9)
    Rl = R + 0.01 * np.real(np.trace(R, axis1=-2, axis2=-1))[:, None, None] / 16 * np.eye(16)
    np.testing.assert_allclose(capon(R, lat, grid, loading=0.01, method=method), 1 / quad(np.linalg.inv(Rl)), rtol=1e-8)
    _, V = np.linalg.eigh(R)
    En = V[..., :14]
    np.testing.assert_allclose(music(R, lat, grid, 2, method), 1 / quad(En @ np.swapaxes(En.conj(), -1, -2)), rtol=1e-6)


def test_capon_and_music_resolve_two_sources():
    lat = Lattice.ula(8)
    truth = np.array([-0.2, 0.15])  # 0.35 apart, beamwidth 2 / N = 0.25
    R = covariance(snapshots(lat, truth))
    grid = Grid.linspace(-1, 1, 801)
    u = grid.axis(0)
    for p in (capon(R, lat, grid), music(R, lat, grid, 2)):
        np.testing.assert_allclose(two_peaks(p, u), truth, atol=0.01)


def test_music_rejects_bad_source_count():
    with pytest.raises(ValueError):
        music(np.eye(4), Lattice.ula(4), Grid.periodic(8), 4)
