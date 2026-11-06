import numpy as np
import pytest

from lagspec import transform as T

RNG = np.random.default_rng(2)
KAPPA = np.pi

AXES = {
    "periodic": (-1.0, 2.0 / 32, 32),
    "periodic_aliasing": (-1.0, 2.0 / 8, 8),  # fewer bins than offsets
    "zoom": (-0.3, 0.01, 41),
}


def direct_matrix(offsets, axis, sign):
    start, step, num = axis
    return np.exp(sign * 1j * KAPPA * np.outer(offsets, start + step * np.arange(num)))


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("sign", [1, -1])
def test_forward_and_adjoint_match_direct(axis, sign):
    ax = AXES[axis]
    offsets = np.arange(-6, 7)
    E = direct_matrix(offsets, ax, sign)
    c = RNG.standard_normal((3, len(offsets))) + 1j * RNG.standard_normal((3, len(offsets)))
    y = RNG.standard_normal((3, ax[2])) + 1j * RNG.standard_normal((3, ax[2]))
    methods = ["matmul", "fft"] if T.is_periodic(KAPPA, ax[1], ax[2]) else ["matmul"]
    for m in methods + ["auto"]:
        np.testing.assert_allclose(T.forward(c, offsets, ax, KAPPA, sign, m), c @ E, atol=1e-10)
        np.testing.assert_allclose(T.adjoint(y, offsets, ax, KAPPA, sign, m), y @ E.T, atol=1e-10)


def test_adjoint_is_the_hermitian_adjoint():
    ax = AXES["zoom"]
    offsets = np.arange(-3, 5)
    c = RNG.standard_normal(len(offsets)) + 1j * RNG.standard_normal(len(offsets))
    y = RNG.standard_normal(ax[2]) + 1j * RNG.standard_normal(ax[2])
    lhs = np.vdot(y, T.forward(c, offsets, ax, KAPPA, +1, "matmul"))
    rhs = np.vdot(T.adjoint(y, offsets, ax, KAPPA, -1, "matmul"), c)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        T.forward(np.ones(3), np.arange(3), AXES["zoom"], KAPPA, 1, "bogus")
