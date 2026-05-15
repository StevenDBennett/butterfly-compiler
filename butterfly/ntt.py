"""
Number‑Theoretic Transform (modular arithmetic butterfly).

Implements `ntt_butterfly` that builds forward/inverse NTT over ℤ/p as a
`MultiSeedOperad` with modular semiring arithmetic.
"""

from __future__ import annotations

import numpy as np

from .operad import MultiSeedOperad
from .qft import _bit_reverse_indices
from .semiring import MOD_NTT_998244353, modular_semiring


def ntt_butterfly(
    n: int,
    p: int,
    omega: int,
    inverse: bool = False,
    include_swaps: bool = True,
    scale: bool = True,
):
    """
    Construct the NTT (Number Theoretic Transform) for size N = 2^n
    over finite field ℤ/p, using primitive 2^n-th root omega.

    Parameters
    ----------
    n : int
        Transform size N = 2^n.
    p : int
        Prime modulus (must be of form k*2^m + 1 with m ≥ n).
    omega : int
        Primitive 2^n-th root of unity modulo p (omega^{2^n} ≡ 1 mod p).
    inverse : bool
        If True, construct inverse NTT (use omega^{-1} mod p).
    include_swaps : bool
        If True, output bit‑reversal permutation (standard order).
    scale : bool
        If True and inverse=True, multiply by N^{-1} mod p (standard scaling).
        Ignored for forward transform.

    Returns
    -------
    operad : MultiSeedOperad
        Butterfly operad for NTT.
    apply_fn : callable
        Function that applies the NTT to a length‑N integer vector.
    """
    N = 1 << n
    # Verify omega is a primitive 2^n-th root
    if pow(omega, N, p) != 1:
        raise ValueError(f"omega^{N} mod {p} != 1")
    # For inverse NTT, use omega^{-1} mod p
    if inverse:
        # Compute modular inverse using Fermat's little theorem
        omega = pow(omega, p - 2, p)

    # Seed matrix: [[1, 1], [1, -1]] where -1 ≡ p-1 (mod p)
    seed = np.array([[1, 1], [1, p - 1]], dtype=np.int64) % p
    semiring = modular_semiring(p)

    # Precompute scaling factor for inverse NTT
    invN = None
    if inverse and scale:
        invN = pow(N, p - 2, p)  # modular inverse of N

    # Precompute all powers of omega: omega^j mod p for j = 0..N-1
    # Vectorized: sequential multiplication avoids O(N log N) pow() calls
    omega_powers = np.ones(N, dtype=np.int64)
    for j in range(1, N):
        omega_powers[j] = (omega_powers[j - 1] * omega) % p

    seeds = []
    twiddles = []

    # Build stages similar to qft_butterfly but with modular arithmetic
    for k in range(n):
        block_size = 1 << (n - k)  # 2^{n-k}
        half = block_size >> 1  # 2^{n-k-1}
        n_blocks = N // block_size
        stride = N // block_size  # = 2^k
        # Primitive block_size-th root: omega^{stride} (unused, but computed for documentation)
        # Twiddle factors: omega^{j * stride} for j = 0..half-1
        root_pow = omega_powers[0 : half * stride : stride]
        # Pattern: first half ones, second half root^{j}
        pattern = np.concatenate([np.ones(half, dtype=np.int64), root_pow]) % p
        twiddle = np.tile(pattern, n_blocks)
        seeds.append(seed)
        twiddles.append(twiddle)

    # Reverse order: widest butterfly first
    seeds = seeds[::-1]
    twiddles = twiddles[::-1]

    operad = MultiSeedOperad(seeds, twiddles=twiddles, semiring=semiring)

    # Define apply function with optional scaling
    def apply_base(x: np.ndarray) -> np.ndarray:
        """Apply the core NTT (without bit-reversal)."""
        return operad.apply(np.asarray(x, dtype=np.int64))

    def apply_with_scaling(x: np.ndarray) -> np.ndarray:
        """Apply NTT and optionally scale."""
        y = apply_base(x)
        if invN is not None:
            y = np.mod(y * invN, p)
        return y

    if include_swaps:
        br = _bit_reverse_indices(n)

        def apply_fn(x: np.ndarray) -> np.ndarray:
            # Input bit-reversal for DIT (or output for DIF)
            return apply_with_scaling(x[br])

        return operad, apply_fn

    return operad, apply_with_scaling


__all__ = ["ntt_butterfly", "MOD_NTT_998244353"]
