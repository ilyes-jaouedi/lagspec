"""
One-axis Fourier series between consecutive integer offsets o = o_0 .. o_0 + I - 1 and a
uniform direction-cosine axis u_k = start + k * step, k = 0 .. K - 1:

    forward   y_k = sum_o c_o exp(sign j kappa o u_k)     (coefficients -> grid)
    adjoint   c_o = sum_k y_k exp(sign j kappa o u_k)     (grid -> coefficients)

Each is computed by a small matrix product, by an FFT when the axis covers one period
of the series (kappa * step * K = 2 pi), or by a chirp-z transform (Bluestein: three
FFTs) on any uniform axis, e.g. a fine zoom on a sub-region with a large array. The
engine applies them one axis at a time on the last axis of an array.
"""

import numpy as np

from . import _backend as B

# method="auto" uses the FFT on a periodic axis when the number of offsets exceeds
# FFT_FACTOR * log2(grid size): below that, the small BLAS product is faster
# (benchmarks/bench_quad.py)
FFT_FACTOR = 8.0
# method="auto" uses the chirp-z transform on a non-periodic axis when
# n_offsets * grid size > CZT_FACTOR * L log2 L, L the FFT length (benchmarks/bench_czt.py)
CZT_FACTOR = 8.0
# largest 0/1 accumulation matrix (inputs x outputs entries) kept dense
DENSE_LIMIT = 4_000_000

METHODS = ("auto", "matmul", "fft", "czt")


def is_periodic(kappa, step, num):
    return bool(np.isclose(kappa * step * num, 2 * np.pi, rtol=1e-9, atol=0.0))


def choose(method, kappa, axis, n_offsets):
    """the evaluation path of one axis"""
    start, step, num = axis
    periodic = is_periodic(kappa, step, num)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "fft":
        if not periodic:
            raise ValueError("method='fft' needs a periodic axis (see Grid.periodic)")
        return "fft"
    if method != "auto":
        return method
    if periodic:
        return "fft" if n_offsets > FFT_FACTOR * np.log2(max(num, 2)) else "matmul"
    L = _fft_length(n_offsets + num - 1)
    return "czt" if n_offsets * num > CZT_FACTOR * L * np.log2(L) else "matmul"


def _fft_length(n):
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def _concat(xp, arrays):
    return np.concatenate(arrays, axis=-1) if xp is np else xp.concat(arrays, axis=-1)


def _pad(xp, x, n):
    """zero-pad the last axis of x to length n"""
    extra = n - x.shape[-1]
    if extra == 0:
        return x
    zeros = xp.zeros(x.shape[:-1] + (extra,), dtype=x.dtype, **_dev(xp, x))
    return _concat(xp, [x, zeros])


def _dev(xp, x):
    return {} if xp is np else {"device": B.device(x)}


def _const(xp, a, like):
    return B.asarray(xp, a, like, dtype=like.dtype)


def _chirp(xp, x, theta, n_out):
    """y_j = sum_i x_i exp(j theta i j), j = 0 .. n_out - 1, along the last axis (Bluestein)"""
    n_in = x.shape[-1]
    L = _fft_length(n_in + n_out - 1)
    i = np.arange(n_in, dtype=float)
    a = _pad(xp, x * _const(xp, np.exp(0.5j * theta * i**2), x), L)
    m = np.arange(-(n_in - 1), n_out)
    b = np.zeros(L, dtype=complex)
    b[m % L] = np.exp(-0.5j * theta * m.astype(float) ** 2)  # i j = (i^2 + j^2 - (j - i)^2) / 2
    conv = xp.fft.ifft(xp.fft.fft(a, axis=-1) * _const(xp, np.fft.fft(b), x), axis=-1)
    j = np.arange(n_out, dtype=float)
    return conv[..., :n_out] * _const(xp, np.exp(0.5j * theta * j**2), x)


def _fft_sign(xp, x, sign):
    """sum_k x_k exp(sign j 2 pi k m / K) for m = 0 .. K - 1, along the last axis"""
    n = x.shape[-1]
    return xp.fft.ifft(x, axis=-1) * n if sign > 0 else xp.fft.fft(x, axis=-1)


def forward(c, offsets, axis, kappa, sign, method):
    """last axis of c (complex): coefficients at `offsets` -> values on the grid axis"""
    start, step, num = axis
    path = choose(method, kappa, axis, len(offsets))
    xp = B.namespace(c)
    if path == "matmul":
        return c @ _const(xp, np.exp(sign * 1j * kappa * np.outer(offsets, start + step * np.arange(num))), c)
    if path == "czt":
        # o = o_0 + i: exp(j kappa o u_k) = exp(j kappa o_0 u_k) exp(j kappa i start) exp(j kappa step i k)
        i = offsets - offsets[0]
        y = _chirp(xp, c * _const(xp, np.exp(sign * 1j * kappa * i * start), c), sign * kappa * step, num)
        return y * _const(xp, np.exp(sign * 1j * kappa * offsets[0] * (start + step * np.arange(num))), c)
    # exp(j kappa o (start + k step)) = exp(j kappa o start) exp(j 2 pi o k / num): the
    # offsets go to bins o mod num (consecutive, so: pad, fold the aliased ones, roll)
    phased = c * _const(xp, np.exp(sign * 1j * kappa * offsets * start), c)
    reps = -(-len(offsets) // num)
    folded = _pad(xp, phased, reps * num)
    if reps > 1:  # offsets that alias onto one bin add up (exact)
        folded = xp.sum(xp.reshape(folded, folded.shape[:-1] + (reps, num)), axis=-2)
    return _fft_sign(xp, xp.roll(folded, int(offsets[0] % num), axis=-1), sign)


def adjoint(y, offsets, axis, kappa, sign, method):
    """last axis of y (complex): values on the grid axis -> coefficients at `offsets`"""
    start, step, num = axis
    path = choose(method, kappa, axis, len(offsets))
    xp = B.namespace(y)
    if path == "matmul":
        return y @ _const(xp, np.exp(sign * 1j * kappa * np.outer(start + step * np.arange(num), offsets)), y)
    if path == "czt":
        x = y * _const(xp, np.exp(sign * 1j * kappa * offsets[0] * step * np.arange(num)), y)
        return _chirp(xp, x, sign * kappa * step, len(offsets)) * _const(xp, np.exp(sign * 1j * kappa * offsets * start), y)
    full = _fft_sign(xp, y, sign)
    idx = B.asarray(xp, offsets % num, y)
    return xp.take(full, idx, axis=-1) * _const(xp, np.exp(sign * 1j * kappa * offsets * start), y)


class Accumulator:
    """
    sums the last axis of an array (n_in entries) into n_out bins, entry i going to bin
    index[i]: one product with a 0/1 matrix when small (always for non-NumPy arrays, the
    matrix cached per namespace, device and dtype), a bincount otherwise
    """

    def __init__(self, index, n_out):
        self.index = np.asarray(index, dtype=np.int64)
        self.n_out = int(n_out)
        n_in = len(self.index)
        self.matrix = None
        if n_in * self.n_out <= DENSE_LIMIT:
            self.matrix = self._dense()
        self._cache = {}

    def _dense(self):
        m = np.zeros((len(self.index), self.n_out))
        m[np.arange(len(self.index)), self.index] = 1.0
        return m

    def __call__(self, x):
        xp = B.namespace(x)
        if xp is not np:
            key = (xp.__name__, str(B.device(x)), str(x.dtype))
            if key not in self._cache:
                m = self.matrix if self.matrix is not None else self._dense()
                self._cache[key] = B.asarray(xp, m, x, dtype=x.dtype)
            flat = xp.reshape(x, (-1, x.shape[-1]))
            return xp.reshape(flat @ self._cache[key], x.shape[:-1] + (self.n_out,))
        x = np.asarray(x)
        batch = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        if self.matrix is not None:
            out = flat @ self.matrix
        else:
            b = len(flat)
            idx = (np.arange(b)[:, None] * self.n_out + self.index[None, :]).ravel()
            size = b * self.n_out
            out = np.bincount(idx, flat.real.ravel(), size)
            if np.iscomplexobj(flat):
                out = out + 1j * np.bincount(idx, flat.imag.ravel(), size)
            out = out.reshape(b, self.n_out)
        return out.reshape(batch + (self.n_out,))
