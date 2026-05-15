"""
Semiring abstraction for the Butterfly Compiler.

Defines the Semiring class and standard semiring instances (real, tropical,
GF(2), log‑domain, modular arithmetic).  The butterfly algorithm works over
any semiring, enabling transforms over diverse algebraic structures.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np


class Semiring:
    """
    Abstract base for semiring operations.

    The butterfly algorithm works over ANY semiring (R, ⊕, ⊗, 0, 1) with:
    - Associativity of ⊕ and ⊗
    - Distributivity of ⊗ over ⊕
    - Identity elements 0 and 1

    This enables: tropical (min,+), GF(2) (XOR,AND), log-semiring, etc.
    """

    def __init__(
        self,
        add: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mul: Callable[[np.ndarray, np.ndarray], np.ndarray],
        zero: float,
        one: float,
        name: str = "abstract",
    ):
        self.add = add
        self.mul = mul
        self.zero = zero
        self.one = one
        self.name = name

    def matvec(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Matrix-vector product A ⊗ x in this semiring.

        Utility method for custom dispatch. Note: fast_kron_power_transform
        inlines its own semiring logic for performance; use this for ad-hoc
        single-level applications or when extending the framework.
        """
        # For 2x2: [a b; c d] ⊗ [u; v] = [a⊗u ⊕ b⊗v; c⊗u ⊕ d⊗v]
        d = A.shape[0]
        h = x.shape[0] // d
        result = np.full_like(x, self.zero)
        for i in range(d):
            for j in range(d):
                result[i * h : (i + 1) * h] = self.add(
                    result[i * h : (i + 1) * h],
                    self.mul(A[i, j], x[j * h : (j + 1) * h]),
                )
        return result

    def __repr__(self) -> str:
        return f"Semiring(name={self.name!r}, zero={self.zero}, one={self.one})"


# Standard semirings
REAL_SEMIRING = Semiring(
    add=lambda a, b: a + b, mul=lambda a, b: a * b, zero=0.0, one=1.0, name="(R,+,×)"
)

TROPICAL_MIN = Semiring(
    add=lambda a, b: np.minimum(a, b),
    mul=lambda a, b: a + b,
    zero=np.inf,
    one=0.0,
    name="(R∪{∞}, min, +)",
)

TROPICAL_MAX = Semiring(
    add=lambda a, b: np.maximum(a, b),
    mul=lambda a, b: a + b,
    zero=-np.inf,
    one=0.0,
    name="(R∪{-∞}, max, +)",
)

GF2_SEMIRING = Semiring(
    add=lambda a, b: (a ^ b) & 1,
    mul=lambda a, b: a & b,
    zero=0,
    one=1,
    name="(GF(2), XOR, AND)",
)

LOG_SEMIRING = Semiring(
    add=lambda a, b: np.logaddexp(a, b),
    mul=lambda a, b: a + b,
    zero=-np.inf,
    one=0.0,
    name="(log-domain, log-sum-exp, +)",
)


@functools.cache
def modular_semiring(p: int) -> Semiring:
    """
    Semiring over integers modulo p (p prime, but any integer works).
    Addition and multiplication are performed modulo p.

    Overflow safety: requires p ≤ 2**31−1 ≈ 2.147e9, so that (p−1)*(p−1) < 2**63−1.
    This holds for typical NTT moduli (e.g., 998244353).
    """
    if not isinstance(p, int) or p <= 1:
        raise ValueError(f"modular_semiring requires p >= 2, got {p}")
    if p > 2**31 - 1:
        raise ValueError(
            f"modular_semiring requires p <= 2**31 - 1 for overflow safety, got {p}"
        )

    def add_mod(a, b):
        return (a + b) % p

    def mul_mod(a, b):
        return (a * b) % p

    semiring = Semiring(
        add=add_mod,
        mul=mul_mod,
        zero=0,
        one=1,
        name=f"(ℤ/{p}, + mod, × mod)",
    )
    semiring.modulus = p
    return semiring


# Common NTT modulus: 998244353 = 119 * 2^23 + 1 (supports N up to 2^23)
MOD_NTT_998244353 = modular_semiring(998244353)


__all__ = [
    "Semiring",
    "REAL_SEMIRING",
    "TROPICAL_MIN",
    "TROPICAL_MAX",
    "GF2_SEMIRING",
    "LOG_SEMIRING",
    "modular_semiring",
    "MOD_NTT_998244353",
]
