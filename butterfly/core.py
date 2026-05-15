"""
Core butterfly transform engine.

Implements the O(N log_d N) Kronecker‑power transform over arbitrary semirings,
together with naïve reference implementations for the Number Theoretic Transform.
"""

from __future__ import annotations

import numpy as np

from .semiring import (
    GF2_SEMIRING,
    LOG_SEMIRING,
    REAL_SEMIRING,
    TROPICAL_MAX,
    TROPICAL_MIN,
    Semiring,
)
from .utils import seed_power

try:
    import numba  # noqa: F401

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


if HAVE_NUMBA:
    import numpy as np
    from numba import njit

    from .semiring import (
        GF2_SEMIRING,
        LOG_SEMIRING,
        REAL_SEMIRING,
        TROPICAL_MAX,
        TROPICAL_MIN,
    )

    # Separate kernels for each semiring to avoid type conflicts
    @njit
    def _kernel_real(seed, blocks, out):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = 0.0
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                src = blocks[j]
                for g in range(n_groups):
                    for k in range(h):
                        out[g, i * h + k] += coeff * src[g, k]
        return out

    @njit
    def _kernel_tropical_min(seed, blocks, out):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        inf = np.inf
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = inf
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                src = blocks[j]
                for g in range(n_groups):
                    for k in range(h):
                        val = coeff + src[g, k]
                        if val < out[g, i * h + k]:
                            out[g, i * h + k] = val
        return out

    @njit
    def _kernel_tropical_max(seed, blocks, out):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        neg_inf = -np.inf
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = neg_inf
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                src = blocks[j]
                for g in range(n_groups):
                    for k in range(h):
                        val = coeff + src[g, k]
                        if val > out[g, i * h + k]:
                            out[g, i * h + k] = val
        return out

    @njit
    def _kernel_gf2(seed, blocks, out):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = 0
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                if coeff:  # coeff == 1
                    src = blocks[j]
                    for g in range(n_groups):
                        for k in range(h):
                            out[g, i * h + k] ^= src[g, k]
        return out

    @njit
    def _kernel_log(seed, blocks, out):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        neg_inf = -np.inf
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = neg_inf
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                src = blocks[j]
                for g in range(n_groups):
                    for k in range(h):
                        a = out[g, i * h + k]
                        b = coeff + src[g, k]
                        if a > b:
                            out[g, i * h + k] = a + np.log1p(np.exp(b - a))
                        else:
                            out[g, i * h + k] = b + np.log1p(np.exp(a - b))
        return out

    @njit
    def _kernel_modular(seed, blocks, out, modulus):
        d = seed.shape[0]
        n_groups = blocks[0].shape[0]
        h = blocks[0].shape[1]
        for g in range(n_groups):
            for i in range(d):
                for k in range(h):
                    out[g, i * h + k] = 0
        for i in range(d):
            for j in range(d):
                coeff = seed[i, j]
                src = blocks[j]
                for g in range(n_groups):
                    for k in range(h):
                        out[g, i * h + k] = (
                            out[g, i * h + k] + coeff * src[g, k]
                        ) % modulus
        return out

    # Map semiring instances to kernels
    _kernel_map = {
        REAL_SEMIRING: _kernel_real,
        TROPICAL_MIN: _kernel_tropical_min,
        TROPICAL_MAX: _kernel_tropical_max,
        GF2_SEMIRING: _kernel_gf2,
        LOG_SEMIRING: _kernel_log,
    }
    # modular_semiring(p) returns different instances; handle by name

    def get_semiring_kernel(semiring):
        """Return (kernel, extra_args) for the given semiring, or None."""
        kernel = _kernel_map.get(semiring)
        extra_args = ()
        if kernel is None:
            # Check if modular semiring
            if semiring.name.startswith("(ℤ/"):
                import re

                match = re.search(r"ℤ/(\d+)", semiring.name)
                if match:
                    modulus = int(match.group(1))
                    kernel = _kernel_modular
                    extra_args = (modulus,)
                else:
                    return None
            else:
                return None
        return (kernel, extra_args)
else:
    get_semiring_kernel = None


def naive_ntt(x: np.ndarray, omega: int, p: int) -> np.ndarray:
    """
    Compute the Number Theoretic Transform of length N = len(x) using
    primitive N-th root omega modulo prime p.
    """
    N = len(x)
    result = np.zeros(N, dtype=np.int64)
    for k in range(N):
        total = 0
        wk = pow(omega, k, p)  # ω^k
        w = 1
        for j in range(N):
            total = (total + x[j] * w) % p
            w = (w * wk) % p  # w = ω^{k*j}
        result[k] = total
    return result


def naive_intt(X: np.ndarray, omega: int, p: int, scale: bool = True) -> np.ndarray:
    """
    Inverse NTT using omega^{-1} mod p.
    If scale=True, multiply by N^{-1} mod p.
    """
    inv_omega = pow(omega, p - 2, p)
    N = len(X)
    result = naive_ntt(X, inv_omega, p)
    if scale:
        invN = pow(N, p - 2, p)
        result = (result * invN) % p
    return result


def fast_kron_power_transform(
    x: np.ndarray,
    seed: np.ndarray,
    semiring: Semiring = REAL_SEMIRING,
    d: int | None = None,
) -> np.ndarray:
    """
    Apply seed^{⊗n} to x in O(N log_d N) over arbitrary semiring.

    Generalized to d×d seeds and arbitrary semirings.
    """
    seed = np.asarray(seed)
    if seed.ndim != 2 or seed.shape[0] != seed.shape[1]:
        raise ValueError(f"seed must be a square 2-D matrix; got shape {seed.shape}")
    if d is None:
        d = seed.shape[0]
    if d != seed.shape[0]:
        raise ValueError(
            f"d={d} doesn't match seed shape {seed.shape[0]}×{seed.shape[1]}"
        )

    # GF(2) semiring requires integer dtypes
    if semiring is GF2_SEMIRING:
        if not np.issubdtype(seed.dtype, np.integer):
            raise TypeError(
                f"GF2 semiring requires integer dtype for seed, got {seed.dtype}"
            )
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(
                f"GF2 semiring requires integer dtype for input, got {x.dtype}"
            )

    x = np.asarray(x).copy()
    N = x.size
    if N == 0:
        raise ValueError("Input vector cannot be empty")

    # Check N is power of d
    temp = N
    while temp > 1:
        if temp % d != 0:
            raise ValueError(f"N={N} must be power of d={d}")
        temp //= d

    # Double-buffering for generic semiring
    src, dest = None, None
    h = 1
    while h < N:
        if semiring is REAL_SEMIRING:
            n_groups = N // (d * h)
            x_reshaped = x.reshape(n_groups, d, h)
            x_new = np.einsum("ij,...jk->...ik", seed, x_reshaped)
            x = x_new.reshape(-1)
        else:
            n_groups = N // (d * h)

            if src is None:
                src = x.copy()
                dest = np.empty_like(x)

            src_reshaped = src.reshape(n_groups, d * h)
            dest_reshaped = dest.reshape(n_groups, d * h)
            blocks = [src_reshaped[:, k * h : (k + 1) * h] for k in range(d)]

            dest_reshaped.fill(semiring.zero)
            if semiring is TROPICAL_MIN:
                for i in range(d):
                    for j in range(d):
                        dest_reshaped[:, i * h : (i + 1) * h] = np.minimum(
                            dest_reshaped[:, i * h : (i + 1) * h],
                            seed[i, j] + blocks[j],
                        )
            elif semiring is TROPICAL_MAX:
                for i in range(d):
                    for j in range(d):
                        dest_reshaped[:, i * h : (i + 1) * h] = np.maximum(
                            dest_reshaped[:, i * h : (i + 1) * h],
                            seed[i, j] + blocks[j],
                        )
            elif semiring is LOG_SEMIRING:
                for i in range(d):
                    for j in range(d):
                        dest_reshaped[:, i * h : (i + 1) * h] = np.logaddexp(
                            dest_reshaped[:, i * h : (i + 1) * h],
                            seed[i, j] + blocks[j],
                        )
            elif semiring is GF2_SEMIRING:
                for i in range(d):
                    for j in range(d):
                        coeff = seed[i, j]
                        if coeff:
                            dest_reshaped[:, i * h : (i + 1) * h] = np.bitwise_xor(
                                dest_reshaped[:, i * h : (i + 1) * h], blocks[j]
                            )
            elif HAVE_NUMBA and get_semiring_kernel is not None:
                kernel_info = get_semiring_kernel(semiring)
                if kernel_info is not None:
                    kernel, extra_args = kernel_info
                    kernel(seed, blocks, dest_reshaped, *extra_args)
                else:
                    for i in range(d):
                        for j in range(d):
                            contrib = semiring.mul(seed[i, j], blocks[j])
                            dest_reshaped[:, i * h : (i + 1) * h] = semiring.add(
                                dest_reshaped[:, i * h : (i + 1) * h], contrib
                            )
            else:
                for i in range(d):
                    for j in range(d):
                        contrib = semiring.mul(seed[i, j], blocks[j])
                        dest_reshaped[:, i * h : (i + 1) * h] = semiring.add(
                            dest_reshaped[:, i * h : (i + 1) * h], contrib
                        )

            src, dest = dest, src
            x = src.ravel()
        h *= d

    return x


def fast_forward(
    x: np.ndarray,
    seed: np.ndarray,
    t: int,
    semiring: Semiring = REAL_SEMIRING,
    d: int | None = None,
) -> np.ndarray:
    """
    Compute v_t = (seed^{⊗n})^t · x = (seed^t)^{⊗n} · x in O(d³ log t) + O(N log N).

    Parameters
    ----------
    x : np.ndarray
        Input vector of length N = d^n.
    seed : np.ndarray
        d×d matrix.
    t : int
        Non‑negative integer exponent.
    semiring : Semiring, optional
        Semiring for the transform (default: real numbers).
    d : int | None, optional
        Arity d (default: seed.shape[0]).

    Returns
    -------
    np.ndarray
        Result vector of same length as x.
    """
    d_seed = seed.shape[0]
    if d is None:
        d = d_seed
    if d != d_seed:
        raise ValueError(f"d={d} does not match seed shape {d_seed}×{d_seed}")
    if semiring is not REAL_SEMIRING:
        raise NotImplementedError(
            "fast_forward currently only supports REAL_SEMIRING; "
            "other semirings require semiring‑aware exponentiation."
        )
    seed_pow = seed_power(seed, t)
    return fast_kron_power_transform(x, seed_pow, semiring, d)


def rule90_fast_forward(x: np.ndarray, t: int) -> np.ndarray:
    """
    Rule 90 cellular automaton over GF(2). Seed S = [[1,1],[1,1]] satisfies S² ≡ 0 mod 2,
    so all even t > log₂ N give the zero state (Wolfram's extinction).

    Parameters
    ----------
    x : np.ndarray
        Input vector of length N = 2^n over GF(2) (0/1 integers).
    t : int
        Non‑negative integer exponent.

    Returns
    -------
    np.ndarray
        Result vector over GF(2) (same length).
    """
    from .semiring import GF2_SEMIRING

    seed = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    seed_pow = seed_power(seed, t) % 2
    return fast_kron_power_transform(x, seed_pow, GF2_SEMIRING, d=2)


def fast_multi_seed(
    x: np.ndarray,
    seeds: list[np.ndarray],
    semiring: Semiring = REAL_SEMIRING,
) -> np.ndarray:
    """
    Butterfly with a different d×d seed at each level (multi‑seed braid).

    Parameters
    ----------
    x : np.ndarray
        Input vector of length N = d_0 * d_1 * … * d_{k-1} where d_k = seeds[k].shape[0].
    seeds : list[np.ndarray]
        List of square matrices; length must equal log_d(N) where d varies per level.
    semiring : Semiring, optional
        Semiring for the transform (default: real numbers).

    Returns
    -------
    np.ndarray
        Result vector of same length as x.

    Notes
    -----
    This is a convenience wrapper around `MultiSeedOperad`. For more control (twiddles,
    caching), use `MultiSeedOperad` directly.
    """
    from .operad import MultiSeedOperad

    return MultiSeedOperad(seeds, semiring=semiring).apply(x)


def fast_tropical_transform(
    x: np.ndarray,
    seed: np.ndarray,
    op: str = "min-plus",
) -> np.ndarray:
    """
    Butterfly over the (min,+) or (max,+) tropical semiring.
    Seed entries are costs; identity is 0, annihilator is ±∞.
    Covers shortest paths, Viterbi, high-SNR polar SC decoder.
    """
    x = np.asarray(x).ravel().copy()
    # Preserve integer dtypes for precision (e.g., combinatorial path lengths)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(float)
    (a, b), (c, d) = seed.astype(float)
    combine = np.minimum if op == "min-plus" else np.maximum
    N, h = x.size, 1
    while h < N:
        x = x.reshape(-1, 2 * h)
        u = x[:, :h].copy()
        v = x[:, h:].copy()
        x[:, :h] = combine(a + u, b + v)
        x[:, h:] = combine(c + u, d + v)
        x = x.ravel()
        h *= 2
    return x


def fast_gf2_transform(
    x: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """
    Butterfly over GF(2). Use cases:
    - Arıkan polar encoder: seed = [[1,0],[1,1]]
    - Rule 90 CA one step: seed = [[1,1],[1,1]]
    - Subset-sum mod 2: seed = [[1,0],[1,1]]
    """
    x = np.asarray(x, dtype=np.uint8).ravel().copy()
    seed_u8 = seed.astype(np.uint8)
    (a, b), (c, d) = seed_u8
    N, h = x.size, 1
    while h < N:
        x = x.reshape(-1, 2 * h)
        u = x[:, :h].copy()
        v = x[:, h:].copy()
        x[:, :h] = (a * u ^ b * v) & 1
        x[:, h:] = (c * u ^ d * v) & 1
        x = x.ravel()
        h *= 2
    return x


__all__ = [
    "naive_ntt",
    "naive_intt",
    "fast_kron_power_transform",
    "fast_forward",
    "rule90_fast_forward",
    "fast_multi_seed",
    "fast_tropical_transform",
    "fast_gf2_transform",
]
