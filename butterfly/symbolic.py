"""
Symbolic fast‑forwarding via polynomial matrices.

Provides `SymbolicMatrix` for representing matrices with polynomial entries,
and functions `symbolic_matrix_power` and `symbolic_kronecker_power` to compute
entries of high ordinary/Kronecker powers without expanding the whole matrix.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_ATOL


class SymbolicMatrix:
    """
    Matrix with polynomial entries, supporting truncated power series.

    Enables computing (S^{⊗n})_{i,j} mod (x^p, y^q,...) in O(n·D) ops
    where D is the number of distinct monomials.

    coeffs: {(row, col, exponent_tuple): coefficient}
    n_vars: number of polynomial variables (inferred from exponent tuple length)
    """

    def __init__(
        self,
        coeffs: dict,
        max_degree: int = 100,
        n_vars: int | None = None,
        shape: tuple[int, int] | None = None,
    ):
        self.coeffs = coeffs
        self.max_degree = max_degree
        # Infer n_vars from first monomial exponent tuple, or default to 2
        if n_vars is not None:
            self.n_vars = n_vars
        elif coeffs:
            self.n_vars = len(next(iter(coeffs))[2])
        else:
            self.n_vars = 2
        if shape is None:
            raise ValueError(
                "shape must be provided; cannot infer from nonzero entries"
            )
        self.shape = shape
        # Validate shape covers all entries
        if self.coeffs:
            max_i = max(k[0] for k in self.coeffs)
            max_j = max(k[1] for k in self.coeffs)
            if max_i >= self.shape[0] or max_j >= self.shape[1]:
                raise ValueError(
                    f"Entry at ({max_i}, {max_j}) exceeds specified shape {self.shape}"
                )

    @classmethod
    def from_matrix(cls, M: np.ndarray, n_vars: int = 2, max_degree: int = 100):
        """
        Create a SymbolicMatrix from a numeric numpy array.
        Each entry becomes a constant polynomial (zero-exponent monomial).
        """
        zero_exp = (0,) * n_vars
        coeffs = {}
        for (i, j), val in np.ndenumerate(M):
            v = complex(val)
            if abs(v) > DEFAULT_ATOL:
                coeffs[(i, j, zero_exp)] = v.real if v.imag == 0 else v
        return cls(coeffs, max_degree=max_degree, n_vars=n_vars, shape=M.shape)

    def __matmul__(self, other: SymbolicMatrix) -> SymbolicMatrix:
        """Matrix multiplication with truncation."""
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        result = {}
        for (i, k1, exp1), c1 in self.coeffs.items():
            for (k2, j, exp2), c2 in other.coeffs.items():
                if k1 != k2:
                    continue
                new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2, strict=True))
                if any(e > self.max_degree for e in new_exp):
                    continue
                key = (i, j, new_exp)
                val = result.get(key, 0) + c1 * c2
                if abs(val) > DEFAULT_ATOL:
                    result[key] = val
        return SymbolicMatrix(
            result,
            self.max_degree,
            n_vars=self.n_vars,
            shape=(self.shape[0], other.shape[1]),
        )

    def __add__(self, other: SymbolicMatrix) -> SymbolicMatrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} + {other.shape}")
        result = self.coeffs.copy()
        for key, c in other.coeffs.items():
            result[key] = result.get(key, 0) + c
        return SymbolicMatrix(
            result, self.max_degree, n_vars=self.n_vars, shape=self.shape
        )

    def __repr__(self) -> str:
        return f"SymbolicMatrix(shape={self.shape}, n_vars={self.n_vars}, max_degree={self.max_degree}, coeffs={len(self.coeffs)} entries)"

    def get_entry(self, i: int, j: int) -> dict:
        """Get polynomial at position (i,j) as {exponent: coefficient}."""
        return {k[2]: v for k, v in self.coeffs.items() if k[0] == i and k[1] == j}


def symbolic_matrix_power(
    S: SymbolicMatrix, n: int, entry: tuple[int, int] = (0, 0)
) -> dict:
    """
    Compute (S^n)_{i,j} — the (i,j) entry of the n-th ordinary matrix power.

    Uses DP over length-n paths in the weighted de Bruijn graph of S:
        (S^n)_{i,j} = sum_{paths i→j of length n} product of edge weights.

    This is NOT the Kronecker power. See `symbolic_kronecker_power` for S^{⊗n}.
    Complexity: O(n · d · D) where D = number of distinct monomials.
    """
    d = S.shape[0]
    i, j = entry
    zero_exp = (0,) * S.n_vars

    # current[pos] = polynomial for paths of current length ending at pos
    current = {pos: {zero_exp: 1.0} if pos == i else {} for pos in range(d)}

    for _step in range(n):
        new_current = {pos: {} for pos in range(d)}
        for pos in range(d):
            for prev in range(d):
                prev_poly = current[prev]
                S_poly = S.get_entry(prev, pos)
                for exp1, c1 in prev_poly.items():
                    if abs(c1) < DEFAULT_ATOL:
                        continue
                    for exp2, c2 in S_poly.items():
                        new_exp = tuple(
                            e1 + e2 for e1, e2 in zip(exp1, exp2, strict=True)
                        )
                        if any(e > S.max_degree for e in new_exp):
                            continue
                        new_current[pos][new_exp] = (
                            new_current[pos].get(new_exp, 0) + c1 * c2
                        )
        current = new_current

    return current[j]


def symbolic_kronecker_power(
    S: SymbolicMatrix, n: int, row_tuple: tuple[int, ...], col_tuple: tuple[int, ...]
) -> dict:
    """
    Compute (S^{⊗n})_{row_tuple, col_tuple} — a single entry of the true Kronecker power.

    For a d×d seed:
        (S^{⊗n})_{(i₁…iₙ), (j₁…jₙ)} = Π_{k=1}^{n} S_{iₖ, jₖ}

    Each factor S_{iₖ,jₖ} is a polynomial; the result is their product
    (with monomial exponents accumulated and truncated at max_degree).

    Complexity: O(n · D) where D = number of distinct monomials.
    """
    if not (len(row_tuple) == len(col_tuple) == n):
        raise ValueError(f"row/col tuples must have length n={n}")
    zero_exp = (0,) * S.n_vars
    result: dict = {zero_exp: 1.0}

    for k in range(n):
        factor = S.get_entry(row_tuple[k], col_tuple[k])
        new_result: dict = {}
        for exp1, c1 in result.items():
            if abs(c1) < DEFAULT_ATOL:
                continue
            for exp2, c2 in factor.items():
                new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2, strict=True))
                if any(e > S.max_degree for e in new_exp):
                    continue
                new_result[new_exp] = new_result.get(new_exp, 0) + c1 * c2
        result = new_result

    return result


__all__ = [
    "SymbolicMatrix",
    "symbolic_matrix_power",
    "symbolic_kronecker_power",
]
