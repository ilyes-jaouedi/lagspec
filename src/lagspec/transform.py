"""
One-axis Fourier series between consecutive integer offsets o = o_0 .. o_0 + I - 1 and a
uniform direction-cosine axis u_k = start + k * step, k = 0 .. K - 1:

    forward   y_k = sum_o c_o exp(sign j kappa o u_k)     (coefficients -> grid)
    adjoint   c_o = sum_k y_k exp(sign j kappa o u_k)     (grid -> coefficients)

Each is computed by a small matrix product, or by an FFT when the axis covers one
period of the series (kappa * step * K = 2 pi). The engine applies them one axis at a
time on the last axis of an array.
"""

import numpy as np

# method="auto" uses the FFT on a periodic axis when the number of offsets exceeds
# FFT_FACTOR * log2(grid size): below that, the small BLAS product is faster
# (benchmarks/bench_quad.py)
FFT_FACTOR = 8.0
# largest 0/1 accumulation matrix (inputs x outputs entries) kept dense
DENSE_LIMIT = 4_000_000

METHODS = ("auto", "matmul", "fft")


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
    if method == "auto" and periodic and n_offsets > FFT_FACTOR * np.log2(max(num, 2)):
        return "fft"
    return "matmul"


def _fft_sign(x, sign, axis=-1):
    """sum_k x_k exp(sign j 2 pi k m / K) for m = 0 .. K - 1"""
    n = x.shape[axis]
    return np.fft.ifft(x, axis=axis) * n if sign > 0 else np.fft.fft(x, axis=axis)


def forward(c, offsets, axis, kappa, sign, method):
    """last axis of c: coefficients at `offsets` -> values on the grid axis"""
    start, step, num = axis
    path = choose(method, kappa, axis, len(offsets))
    if path == "matmul":
        return c @ np.exp(sign * 1j * kappa * np.outer(offsets, start + step * np.arange(num)))
    # exp(j kappa o (start + k step)) = exp(j kappa o start) exp(j 2 pi o k / num)
    phased = c * np.exp(sign * 1j * kappa * offsets * start)
    out = np.zeros(c.shape[:-1] + (num,), dtype=complex)
    bins = offsets % num
    if len(offsets) <= num:  # consecutive offsets: distinct bins
        out[..., bins] = phased
    else:
        for j, k in enumerate(bins):  # offsets that alias onto one bin add up (exact)
            out[..., k] += phased[..., j]
    return _fft_sign(out, sign)


def adjoint(y, offsets, axis, kappa, sign, method):
    """last axis of y: values on the grid axis -> coefficients at `offsets`"""
    start, step, num = axis
    path = choose(method, kappa, axis, len(offsets))
    if path == "matmul":
        return y @ np.exp(sign * 1j * kappa * np.outer(start + step * np.arange(num), offsets))
    full = _fft_sign(np.asarray(y, dtype=complex), sign)
    return full[..., offsets % num] * np.exp(sign * 1j * kappa * offsets * start)


class Accumulator:
    """
    sums the last axis of an array (n_in entries) into n_out bins, entry i going to bin
    index[i]: one product with a 0/1 matrix when small, a bincount otherwise
    """

    def __init__(self, index, n_out):
        self.index = np.asarray(index, dtype=np.int64)
        self.n_out = int(n_out)
        n_in = len(self.index)
        self.matrix = None
        if n_in * self.n_out <= DENSE_LIMIT:
            self.matrix = np.zeros((n_in, self.n_out))
            self.matrix[np.arange(n_in), self.index] = 1.0

    def __call__(self, x):
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
