"""
lagspec: fast covariance-based spatial spectra for arrays on a lattice.

Every spectrum of the form a(u)^H M a(u) (Bartlett, Capon, MUSIC, ...) is evaluated on
a whole direction grid through the lag sums of M over the difference co-array.
"""

from .grid import Grid, azel
from .lattice import Lattice
from .methods import bartlett, capon, covariance, music

__version__ = "0.1.0"

__all__ = ["Grid", "Lattice", "azel", "bartlett", "capon", "covariance", "music", "__version__"]
