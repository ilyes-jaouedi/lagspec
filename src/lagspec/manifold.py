"""
Calibrated array manifolds by manifold separation: the steering vectors are expanded on
spatial harmonics,

    a(u) = G h(u),   h_q(u) = exp(j kappa q . u),   q on an integer box

with G an (N, H) matrix fitted from a measured steering table. An ideal lattice is the
case where row n of G has a single non-zero entry at q = p_n (its gain). Every form
reduces to the same form on the harmonics, a lattice:

    a^H M a = h^H (G^H M G) h,   a^H y = h^H (G^H y),   sum_k p_k a a^H = G (sum_k p_k h h^H) G^H

so the lag-domain engine applies unchanged, and angle-dependent element errors stay
exact as long as they are captured by the harmonics.
"""

import numpy as np

from .lattice import Lattice

# largest (N * H)^2 linear system of the exact fit; above it, alternating least squares
EXACT_FIT_LIMIT = 4_000_000


def _box(lo, hi):
    """all integer points of the box lo..hi (inclusive), (H, D)"""
    axes = [np.arange(a, b + 1) for a, b in zip(lo, hi)]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=-1)


class Manifold:
    """
    harmonics  (H,) or (H, D) integer harmonic positions q
    G          (N, H) complex expansion coefficients, a(u) = G h(u)
    spacing    lattice step in wavelengths (sets kappa, as for Lattice)

    Usually built by Manifold.fit (from a measured table) or Manifold.from_lattice.
    Accepted by every method in place of a Lattice.
    """

    def __init__(self, harmonics, G, spacing=0.5):
        self.harmonics = Lattice(harmonics, spacing)
        self.G = np.asarray(G, dtype=complex)
        if self.G.ndim != 2 or self.G.shape[1] != self.harmonics.n_elements:
            raise ValueError("G must have shape (N, number of harmonics)")

    # -- constructors ---------------------------------------------------------------

    @classmethod
    def from_lattice(cls, lattice, extra=0):
        """
        the ideal manifold of a lattice (with its gains) on the box of its positions
        widened by `extra` harmonics on each side, the extra coefficients zero
        """
        q = _box(lattice.positions.min(0) - extra, lattice.positions.max(0) + extra)
        index = {tuple(x): i for i, x in enumerate(q)}
        G = np.zeros((lattice.n_elements, len(q)), dtype=complex)
        gains = np.ones(lattice.n_elements) if lattice.gains is None else lattice.gains
        for n, p in enumerate(lattice.positions):
            G[n, index[tuple(p)]] = gains[n]
        return cls(q, G, lattice.spacing)

    @classmethod
    def fit(cls, table, points, lattice, extra=1, ridge=1e-2, scale=True, n_iter=50, exact=True):
        """
        fit a manifold to a measured steering table

        table    (P, N) measured steering vectors, one row per direction
        points   (P, D) their direction cosines
        lattice  the nominal array (positions, spacing, optional gains): the prior
        extra    harmonics added on each side of the position box
        ridge    relative ridge pulling G towards the nominal lattice, so that the extra
                 harmonics only take what the data ask for
        scale    a free complex factor per direction (the reflector, element pattern
                 and phase reference of each measurement): only the direction of each
                 measured vector is fitted. Solved exactly, as the linear problem
                     min_G sum_p |P_p G h_p|^2 + lam |G - G0|^2,
                 P_p the projector orthogonal to the measured vector p (the off-model
                 energy, linear in G); for very large N * H (or exact=False), by
                 alternating G and the factors (n_iter rounds, slower to converge).
        """
        prior = cls.from_lattice(lattice, extra)
        table = np.asarray(table, dtype=complex)
        points = np.asarray(points, dtype=float).reshape(len(table), -1)
        H = prior.harmonics.steering(points)  # (P, H)
        G0 = prior.G
        n, nh = G0.shape
        if scale and exact and (n * nh) ** 2 <= EXACT_FIT_LIMIT:
            t = table / np.linalg.norm(table, axis=1, keepdims=True)
            proj = np.eye(n) - t[:, :, None] * t.conj()[:, None, :]  # (P, N, N)
            # sum_p |P_p (h_p^T kron I) vec(G)|^2, vec(G)[a * N + i] = G[i, a]
            A = np.einsum("pa,pb,pij->aibj", H.conj(), H, proj, optimize=True).reshape(n * nh, n * nh)
            lam = ridge * np.real(np.trace(A)) / len(A)
            g = np.linalg.solve(A + lam * np.eye(len(A)), lam * G0.T.reshape(-1))
            G = g.reshape(nh, n).T
            return cls(prior.harmonics.positions, G * np.linalg.norm(G0) / np.linalg.norm(G), lattice.spacing)
        G = G0.copy()
        alpha = np.ones(len(table), dtype=complex)
        for _ in range(n_iter if scale else 1):
            if scale:
                B = H @ G.T  # (P, N) current model
                alpha = np.sum(B.conj() * table, 1) / np.maximum(np.sum(np.abs(B) ** 2, 1), 1e-300)
            D = alpha[:, None] * H
            lhs = D.conj().T @ D
            lam = ridge * np.real(np.trace(lhs)) / lhs.shape[0]
            G = np.linalg.solve(lhs + lam * np.eye(lhs.shape[0]), D.conj().T @ table + lam * G0.T).T
            if scale:  # the overall scale is shared by G and alpha: keep G at the prior's
                G *= np.linalg.norm(G0) / np.linalg.norm(G)
        return cls(prior.harmonics.positions, G, lattice.spacing)

    # -- properties -----------------------------------------------------------------

    @property
    def n_elements(self):
        return self.G.shape[0]

    @property
    def ndim(self):
        return self.harmonics.ndim

    @property
    def spacing(self):
        return self.harmonics.spacing

    @property
    def kappa(self):
        return self.harmonics.kappa

    @property
    def gains(self):
        return None

    # -- forms ----------------------------------------------------------------------

    def quad(self, M, grid, method="auto", real=True):
        """a(u)^H M a(u) on the grid, (..., N, N) -> (..., *grid.shape)"""
        M = np.asarray(M)
        return self.harmonics.quad(self.G.conj().T @ M @ self.G, grid, method, real)

    def linear(self, y, grid, method="auto"):
        """a(u)^H y on the grid, (..., N) -> (..., *grid.shape)"""
        return self.harmonics.linear(np.asarray(y) @ self.G.conj(), grid, method)

    def covariance_from_spectrum(self, p, grid, method="auto"):
        """sum_k p_k a(u_k) a(u_k)^H, (..., *grid.shape) -> (..., N, N)"""
        Rh = self.harmonics.covariance_from_spectrum(p, grid, method)
        return self.G @ Rh @ self.G.conj().T

    def steering(self, points):
        """(P, D) direction cosines -> (P, N) steering vectors"""
        return self.harmonics.steering(points) @ self.G.T

    def off_model(self, table, points):
        """
        per measured direction, the fraction of the measured vector's energy outside the
        model's steering vector (scale-free): 1 - |b^H v|^2 / (|b|^2 |v|^2). Capon cannot
        show a peak-to-floor above about 1 / off_model.
        """
        table = np.asarray(table, dtype=complex)
        B = self.steering(points)
        num = np.abs(np.sum(B.conj() * table, 1)) ** 2
        den = np.sum(np.abs(B) ** 2, 1) * np.sum(np.abs(table) ** 2, 1)
        return 1.0 - num / den

    # -- direct reference -----------------------------------------------------------

    def quad_direct(self, M, grid, real=True):
        A = self.steering(grid.points())
        M = np.asarray(M)
        q = np.einsum("pi,...ij,pj->...p", A.conj(), M, A, optimize=True).reshape(M.shape[:-2] + grid.shape)
        return q.real if real else q

    def linear_direct(self, y, grid):
        y = np.asarray(y)
        return (y @ self.steering(grid.points()).conj().T).reshape(y.shape[:-1] + grid.shape)

    def covariance_from_spectrum_direct(self, p, grid):
        p = np.asarray(p)
        A = self.steering(grid.points())
        flat = p.reshape(p.shape[: p.ndim - self.ndim] + (-1,))
        return np.einsum("...p,pi,pj->...ij", flat, A, A.conj(), optimize=True)

    def __repr__(self):
        return f"Manifold({self.n_elements} elements, {self.G.shape[1]} harmonics, {self.ndim}-D)"
