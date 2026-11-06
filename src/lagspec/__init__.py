"""
lagspec: fast covariance-based spatial spectra for arrays on a lattice.

Every spectrum of the form a(u)^H M a(u) (Bartlett, Capon, MUSIC, ...) is evaluated on
a whole direction grid through the lag sums of M over the difference co-array; IAA
also builds its model covariance through the lags.
"""

from .grid import Grid, azel
from .lattice import Lattice
from .methods import bartlett, capon, covariance, iaa, music

__version__ = "0.2.0"

__all__ = ["Grid", "Lattice", "azel", "bartlett", "capon", "covariance", "iaa", "music", "__version__"]
