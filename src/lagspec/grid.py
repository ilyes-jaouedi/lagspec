"""Evaluation grids, uniform along each direction-cosine axis."""

import numpy as np


class Grid:
    """
    A separable grid of directions: along axis d, the direction cosines
    u_d[k] = start_d + k * step_d, k = 0 .. num_d - 1.

    For a planar array in the x-z plane, axis 0 is u = cos(el) sin(az) and axis 1 is
    v = sin(el) (see `azel`). A grid uniform in direction cosines is what makes the
    lag-domain evaluation exact; it is not uniform in angle.
    """

    def __init__(self, axes):
        self.params = [(float(s), float(h), int(n)) for s, h, n in axes]
        if any(n < 1 for _, _, n in self.params):
            raise ValueError("every axis needs at least one point")

    @classmethod
    def periodic(cls, num, spacing=0.5):
        """
        One full period of the array response per axis: u in [-P/2, P/2) with
        P = 1 / spacing (u in [-1, 1) for half-wavelength spacing). Exact for the
        FFT path; points with |u| > 1 (or u^2 + v^2 > 1) are not physical directions,
        see `visible`.
        """
        nums = np.atleast_1d(num)
        period = 1.0 / spacing
        return cls([(-period / 2, period / n, n) for n in nums])

    @classmethod
    def linspace(cls, start, stop, num):
        """num points from start to stop inclusive, per axis (scalars or sequences)"""
        start, stop, num = np.broadcast_arrays(np.atleast_1d(start), np.atleast_1d(stop), np.atleast_1d(num))
        axes = []
        for a, b, n in zip(start, stop, num):
            step = (b - a) / (n - 1) if n > 1 else 0.0
            axes.append((a, step, n))
        return cls(axes)

    @property
    def ndim(self):
        return len(self.params)

    @property
    def shape(self):
        return tuple(n for _, _, n in self.params)

    def axis(self, d):
        start, step, num = self.params[d]
        return start + step * np.arange(num)

    def mesh(self):
        """one array per axis, broadcast to the grid shape (matrix indexing)"""
        return np.meshgrid(*[self.axis(d) for d in range(self.ndim)], indexing="ij")

    def points(self):
        """(P, D) direction cosines, in the grid's C order"""
        return np.stack([m.ravel() for m in self.mesh()], axis=-1)

    def visible(self):
        """True where the direction cosines describe a real direction (norm <= 1)"""
        return sum(m**2 for m in self.mesh()) <= 1.0

    def __repr__(self):
        axes = ", ".join(f"({s:g}, {h:g}, {n})" for s, h, n in self.params)
        return f"Grid([{axes}])"


def azel(u, v):
    """
    (u, v) = (cos(el) sin(az), sin(el)) -> (azimuth, elevation) in degrees; NaN outside
    the visible region
    """
    u, v = np.asarray(u, float), np.asarray(v, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.arcsin(v)
        az = np.arcsin(u / np.cos(el))
        bad = u**2 + v**2 > 1.0
    return np.where(bad, np.nan, np.degrees(az)), np.where(bad, np.nan, np.degrees(el))
