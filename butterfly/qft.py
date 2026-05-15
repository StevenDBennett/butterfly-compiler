"""
Quantum Fourier Transform via Van‑Loan recursive factorization.

Implements `qft_butterfly` that builds the DFT as a `MultiSeedOperad` with
per‑level twiddle diagonals, and provides bit‑reversal utilities.
"""

from __future__ import annotations

import functools

import numpy as np

from .operad import MultiSeedOperad


@functools.cache
def _bit_reverse_indices(n_bits: int) -> np.ndarray:
    """
    Return array `br` of length 2^n_bits where br[i] = bit-reversal of i.

    Vectorized: O(n_bits) numpy passes, no Python loop over N elements.
    Exposed as a public function for independent testing.
    """
    N = 1 << n_bits
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    tmp = indices.copy()
    for _ in range(n_bits):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1
    return rev


def _bit_reverse_permutation(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Apply bit-reversal permutation to array x."""
    return x[_bit_reverse_indices(n_bits)]


def qft_butterfly(
    n_qubits: int,
    include_swaps: bool = True,
    return_operad: bool = False,
    inverse: bool = False,
):
    """
    Construct the N = 2^n DFT using the Van Loan / Camps–Van Beeumen–Yang
    recursive factorization:

        F'_n = (I_2 ⊗ F'_{n-1}) · (I ⊕ Ω_{n-1}) · (H ⊗ I_{n-1})

    Unrolled into a MultiSeedOperad with per-level twiddle diagonals.
    The operad produces a bit-reversed output (DIT order: bit-reversed input,
    natural output). If include_swaps is True, the input is bit-reversed
    before the butterfly, yielding natural-order output matching
    np.fft.fft(x, norm="ortho").

    Key insight: The QFT breaks the uniform-seed structure of WHT precisely
    because the twiddle diagonal Ω_{n-k-1} is level- AND position-dependent.
    WHT has trivial twiddles (all ones) — that is why it naturally uniforms
    as a Kronecker power and is the correct kernel for pre-conditioning.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (transform size N = 2^{n_qubits}).
    include_swaps : bool, optional
        Whether to include final bit-reversal swaps. Default True.
    return_operad : bool, optional
        If True, return (operad, apply_fn) tuple. If False, return apply_fn.
    inverse : bool, optional
        If True, construct the inverse QFT (conjugated twiddle factors).
        Default False.

    Returns
    -------
    callable or tuple
        apply_fn — callable applying QFT (or inverse QFT) to a vector.
        If return_operad is True, returns (operad, apply_fn).
    """
    n = n_qubits
    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

    seeds = [H] * n

    if inverse:
        twiddles = [np.conj(t) for t in _compute_qft_twiddles(n)]
    else:
        twiddles = _compute_qft_twiddles(n)

    operad = MultiSeedOperad(seeds, twiddles=twiddles)

    if include_swaps:
        br = _bit_reverse_indices(n)

        def apply_fn(x: np.ndarray) -> np.ndarray:
            # DIT: bit-reverse INPUT, then apply ascending-h butterfly stages.
            # The output is already in natural order.
            return operad.apply(np.asarray(x)[br])

    else:
        apply_fn = operad.apply

    if return_operad:
        return operad, apply_fn
    return apply_fn


@functools.cache
def _compute_qft_twiddles(n_qubits: int) -> list[np.ndarray]:
    """
    Compute twiddle diagonals for QFT with n_qubits.
    Returns list of length n_qubits (already reversed for DIT order).
    """
    n = n_qubits
    N = 1 << n
    twiddles = []
    for k in range(n):
        block_size = 1 << (n - k)  # 2^{n-k}
        half = block_size >> 1  # 2^{n-k-1}
        n_blocks = N // block_size
        j_idx = np.arange(half)
        omega_k = np.exp(-2j * np.pi * j_idx / block_size)
        pattern = np.concatenate([np.ones(half, dtype=complex), omega_k])
        twiddle = np.tile(pattern, n_blocks)
        twiddles.append(twiddle)
    # Reverse for DIT order (widest butterfly first)
    return twiddles[::-1]


def qft_direct(x: np.ndarray, include_swaps: bool = True) -> np.ndarray:
    """
    Compute QFT directly using optimized loops.

    This is a specialized implementation that avoids the generic
    MultiSeedOperad overhead, providing better performance.
    """
    x = np.asarray(x, dtype=complex)
    n = int(np.round(np.log2(len(x))))
    if len(x) != (1 << n):
        raise ValueError(f"Length {len(x)} must be power of two")

    if include_swaps:
        br = _bit_reverse_indices(n)
        x = x[br]

    # Precompute twiddles
    twiddles = _compute_qft_twiddles(n)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # Allocate buffers
    buf0 = x
    buf1 = np.empty_like(x)

    h = 1
    for k in range(n):
        tw = twiddles[k]
        if tw is not None:
            buf0 *= tw

        # Reshape to (N//(2*h), 2*h)
        src = buf0.reshape(-1, 2 * h)
        dst = buf1.reshape(-1, 2 * h)

        # Apply Hadamard to each pair of blocks using fused operations
        # dst[:, :h] = (src[:, :h] + src[:, h:]) * inv_sqrt2
        # dst[:, h:] = (src[:, :h] - src[:, h:]) * inv_sqrt2
        np.add(src[:, :h], src[:, h:], out=dst[:, :h])
        np.subtract(src[:, :h], src[:, h:], out=dst[:, h:])
        dst[:, :h] *= inv_sqrt2
        dst[:, h:] *= inv_sqrt2

        # Swap buffers
        buf0, buf1 = buf1, buf0
        h <<= 1

    return buf0


@functools.cache
def _compute_qft_stage_factors(n_qubits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Compute twiddle factors for each stage of QFT.
    Returns list of (tw0, tw1) where tw0 is array of ones (length h)
    and tw1 is exp(-2πi·j/(2h)) for j=0..h-1, for each stage h.
    """
    n = n_qubits
    factors = []
    h = 1
    for _ in range(n):
        # tw0 is all ones (we'll use None to indicate skip multiplication)
        tw1 = np.exp(-2j * np.pi * np.arange(h) / (2 * h))
        factors.append((None, tw1))
        h <<= 1
    return factors


def qft_direct_combined(x: np.ndarray, include_swaps: bool = True) -> np.ndarray:
    """
    Compute QFT with combined twiddle & butterfly operations.
    """
    x = np.asarray(x, dtype=complex)
    n = int(np.round(np.log2(len(x))))
    if len(x) != (1 << n):
        raise ValueError(f"Length {len(x)} must be power of two")

    if include_swaps:
        br = _bit_reverse_indices(n)
        x = x[br]

    factors = _compute_qft_stage_factors(n)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # Allocate buffers
    buf0 = x
    buf1 = np.empty_like(x)

    h = 1
    for k in range(n):
        _, tw1 = factors[k]
        # Reshape to (N//(2*h), 2*h)
        src = buf0.reshape(-1, 2 * h)
        dst = buf1.reshape(-1, 2 * h)

        # Apply twiddle factors and Hadamard in one pass
        # For each group, first half (a) gets tw0 = 1, second half (b) gets tw1
        a = src[:, :h]
        b = src[:, h:]
        # Multiply b by tw1 (broadcast across groups)
        b_tw = b * tw1  # shape (groups, h)

        # Compute butterfly with scaling
        # dst0 = (a + b_tw) * inv_sqrt2
        # dst1 = (a - b_tw) * inv_sqrt2
        np.add(a, b_tw, out=dst[:, :h])
        np.subtract(a, b_tw, out=dst[:, h:])
        dst[:, :h] *= inv_sqrt2
        dst[:, h:] *= inv_sqrt2

        # Swap buffers
        buf0, buf1 = buf1, buf0
        h <<= 1

    return buf0


def fft_seeds(n: int) -> list[np.ndarray]:
    """
    Cooley–Tukey DFT_{2^n} as a multi‑seed butterfly BRAID.

    The FFT is NOT a Kronecker power of a single 2×2 seed; it requires
    level‑dependent twiddle factors. This builder returns the n seeds
    that must be passed to `fast_multi_seed()` or `butterfly_forward()`.

    Parameters
    ----------
    n : int
        Number of levels (size N = 2^n).

    Returns
    -------
    list[np.ndarray]
        List of n complex 2×2 seeds.
    """
    seeds = []
    for k in range(n):
        w = np.exp(2j * np.pi / (2 ** (k + 1)))
        seeds.append(
            (1.0 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, w]], dtype=complex)
        )
    return seeds


__all__ = [
    "qft_butterfly",
    "qft_direct",
    "qft_direct_combined",
    "fft_seeds",
]
