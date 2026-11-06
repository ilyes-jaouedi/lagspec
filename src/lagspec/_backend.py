"""
Array namespace of the inputs: NumPy for NumPy arrays and Python objects (no extra
dependency), otherwise the array-API namespace given by array-api-compat (CuPy,
PyTorch, JAX, ...). Constants of the engine are built with NumPy and moved to the
inputs' namespace, device and precision.
"""

import numpy as np


def namespace(*xs):
    others = [x for x in xs if x is not None and not isinstance(x, (np.ndarray, np.generic, int, float, complex, list, tuple))]
    if not others:
        return np
    try:
        from array_api_compat import array_namespace, is_numpy_namespace
    except ImportError as e:  # pragma: no cover
        raise ImportError("arrays other than NumPy need array-api-compat: pip install lagspec[array-api]") from e
    xp = array_namespace(*others)
    return np if is_numpy_namespace(xp) else xp


def device(x):
    if isinstance(x, np.ndarray):
        return None
    from array_api_compat import device as _device

    return _device(x)


def complex_dtype(xp, x):
    """complex64 for single-precision inputs, complex128 otherwise"""
    single = x.dtype in (xp.float32, xp.complex64)
    return xp.complex64 if single else xp.complex128


def real_dtype(xp, x):
    return xp.float32 if x.dtype in (xp.float32, xp.complex64) else xp.float64


def asarray(xp, a, like, dtype=None):
    """a NumPy constant (or array) in the namespace and on the device of `like`"""
    if xp is np:
        return np.asarray(a, dtype=dtype)
    if not isinstance(a, np.ndarray) or a.dtype != np.dtype(object):
        a = np.ascontiguousarray(a)
    return xp.asarray(a, dtype=dtype, device=device(like))
