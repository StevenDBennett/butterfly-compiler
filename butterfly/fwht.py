"""
Fast Walsh–Hadamard Transform (FWHT) with natural, Paley, and sequency orderings,
and dyadic (XOR) convolution.
"""

from __future__ import annotations

import numpy as np

from .core import fast_kron_power_transform
from .thermodynamics import H2


def _gray_code_indices(n_bits: int) -> np.ndarray:
    """
    Return Gray code sequence of length 2^n_bits.

    The Gray code of integer k is k ^ (k >> 1). This ordering appears in the
    Fino–Algazi sequency‑ordered Walsh–Hadamard transform.

    Parameters
    ----------
    n_bits : int
        Number of bits.

    Returns
    -------
    np.ndarray
        Array of length 2^n_bits containing Gray(k) for k = 0 … 2^n_bits‑1.
    """
    k = np.arange(1 << n_bits, dtype=np.int64)
    return k ^ (k >> 1)


def sequency_permutation(n_bits: int) -> np.ndarray:
    """
    Return permutation that maps position k → Hadamard index bit_reverse(Gray(k)).

    Equivalent to Fino–Algazi's [BI][BR] unscrambler as a single bit‑trick.
    This permutation converts the natural (Kronecker‑power) Hadamard ordering
    into sequency (Walsh) ordering.

    Parameters
    ----------
    n_bits : int
        Number of bits.

    Returns
    -------
    np.ndarray
        Permutation array of length 2^n_bits.
    """
    from .qft import _bit_reverse_indices

    return _bit_reverse_indices(n_bits)[_gray_code_indices(n_bits)]


def fwht(
    x: np.ndarray,
    order: str = "sequency",
    normalize: bool = True,
) -> np.ndarray:
    """
    Unitary‑normalized Fast Walsh–Hadamard Transform.

    Parameters
    ----------
    x : np.ndarray
        Input vector of length N = 2^n.
    order : {"natural", "paley", "sequency"}, optional
        Ordering of the Walsh–Hadamard basis (default "sequency").
    normalize : bool, optional
        If True, use unitary normalization (1/√2 per butterfly).
        If False, use unnormalized transform (1 per butterfly).

    Returns
    -------
    np.ndarray
        Transformed vector of same length as x.

    Raises
    ------
    ValueError
        If input length is not a power of two, or unknown order.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    n = int(round(np.log2(N)))
    if (1 << n) != N:
        raise ValueError(f"Input length {N} is not a power of 2")

    seed = H2 if normalize else np.array([[1.0, 1.0], [1.0, -1.0]])
    y = fast_kron_power_transform(x, seed, d=2)

    if order == "natural":
        return y
    if order == "paley":
        from .qft import _bit_reverse_indices

        return y[_bit_reverse_indices(n)]
    if order == "sequency":
        return y[sequency_permutation(n)]
    raise ValueError(
        f"Unknown order '{order}'; choose 'natural', 'paley', or 'sequency'"
    )


def xor_convolve(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Dyadic (XOR) convolution via WHT pointwise multiplication.

    For vectors f, g of length N = 2^n, compute h where
        h[k] = ∑_{i xor j = k} f[i] * g[j].
    Complexity O(N log N).

    Parameters
    ----------
    f, g : np.ndarray
        Input vectors of same power‑of‑two length.

    Returns
    -------
    np.ndarray
        XOR convolution of f and g.

    Raises
    ------
    ValueError
        If lengths differ or are not powers of two.
    """
    N = f.size
    if g.size != N:
        raise ValueError(f"Input lengths differ: {f.size} vs {g.size}")
    if N & (N - 1):
        raise ValueError(f"Length {N} is not a power of two")

    F = fwht(f, order="natural", normalize=True)
    G = fwht(g, order="natural", normalize=True)
    return np.sqrt(N) * fwht(F * G, order="natural", normalize=True)
