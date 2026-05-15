"""
Block‑butterfly hierarchical compilation.

Implements `BlockSeed` (seed whose entries are matrices) and the corresponding
two‑scale butterfly transform that respects arbitrary semirings.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_TOL
from .semiring import GF2_SEMIRING, REAL_SEMIRING, Semiring
from .solvability import solvability_series


class BlockSeed:
    """
    A seed whose entries are themselves matrices (block matrix).

    Enables recursive compilation: if each block is butterfly-compilable,
    the full block-butterfly is too, with multiplicative depth.
    """

    def __init__(self, blocks: list[list[np.ndarray]]):
        """
        blocks[i][j] is the (i,j)-th block.
        All blocks must be square and same size.
        """
        self.d = len(blocks)
        self.block_size = blocks[0][0].shape[0]
        self.blocks = [[np.asarray(b) for b in row] for row in blocks]
        self.total_size = self.d * self.block_size

    def to_full_matrix(self) -> np.ndarray:
        """Convert to full (d*k) × (d*k) matrix."""
        return np.block(self.blocks)

    def is_butterfly_compilable(self, tol: float = DEFAULT_TOL) -> bool:
        """Check if all blocks are butterfly-compilable."""
        for row in self.blocks:
            for b in row:
                result = solvability_series(b, tol=tol)
                if not result["is_solvable"]:
                    return False
        return True

    def __repr__(self) -> str:
        return f"BlockSeed(d={self.d}, block_size={self.block_size}, total_size={self.total_size})"


def _semiring_block_matmul(
    semiring: Semiring, B: np.ndarray, atoms: np.ndarray
) -> np.ndarray:
    """
    Apply k×k matrix B to batched k-vector atoms using semiring arithmetic.

    atoms: shape (n_groups, h, k)
    B:     shape (k, k)
    Returns: shape (n_groups, h, k)

    In the real semiring this reduces to atoms @ B.T (standard matmul).
    Over other semirings the inner product uses ⊕ and ⊗ as defined.
    """
    k = B.shape[0]
    if semiring is REAL_SEMIRING:
        # Real semiring fast path: standard matrix multiplication
        return atoms @ B.T
    result = np.full_like(atoms, semiring.zero)
    for i in range(k):
        for j_idx in range(k):
            result[:, :, i] = semiring.add(
                result[:, :, i], semiring.mul(B[i, j_idx], atoms[:, :, j_idx])
            )
    return result


def block_butterfly_transform(
    x: np.ndarray, block_seed: BlockSeed, semiring: Semiring = REAL_SEMIRING
) -> np.ndarray:
    """
    True two-scale butterfly with block-matrix seed.

    Input is treated as (N/k) atoms of size k.  The outer d-ary butterfly
    mixes atoms using the block seed's k×k sub-matrices as the "scalar"
    multiplication — implemented via `_semiring_block_matmul` so all
    semirings (real, tropical, GF2, log) are respected.

    Complexity: O((N/k) · log_d(N/k) · d² · k²) = O(N log N · d² · k).
    For constant d and k this is O(N log N).

    N must equal k · d^m for some integer m ≥ 1.
    """
    d = block_seed.d
    k = block_seed.block_size
    N = x.size

    # ── validation ──────────────────────────────────────────────────────────
    if N % k != 0:
        raise ValueError(f"N={N} must be divisible by block_size k={k}")
    # Validate all blocks are k×k
    for row in block_seed.blocks:
        for b in row:
            if b.shape != (k, k):
                raise ValueError(f"All blocks must be {k}×{k}; got {b.shape}")
    # GF(2) semiring requires integer dtypes
    if semiring is GF2_SEMIRING:
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(
                f"GF2 semiring requires integer dtype for input, got {x.dtype}"
            )
        for row in block_seed.blocks:
            for b in row:
                if not np.issubdtype(b.dtype, np.integer):
                    raise TypeError(
                        f"GF2 semiring requires integer dtype for blocks, got {b.dtype}"
                    )
    n_atoms = N // k
    temp = n_atoms
    while temp > 1:
        if temp % d != 0:
            raise ValueError(f"N/k={n_atoms} must be a power of outer arity d={d}")
        temp //= d

    # ── transform ────────────────────────────────────────────────────────────
    X = x.reshape(n_atoms, k).copy()

    h = 1
    while h < n_atoms:
        n_groups = n_atoms // (d * h)
        X_r = X.reshape(n_groups, d, h, k)
        macro_blocks = [X_r[:, j, :, :] for j in range(d)]

        new_X_r = np.full_like(X_r, semiring.zero)
        for i in range(d):
            for j in range(d):
                B_ij = block_seed.blocks[i][j]
                contrib = _semiring_block_matmul(semiring, B_ij, macro_blocks[j])
                new_X_r[:, i, :, :] = semiring.add(new_X_r[:, i, :, :], contrib)

        X = new_X_r.reshape(n_atoms, k)
        h *= d

    return X.ravel()


__all__ = ["BlockSeed", "block_butterfly_transform"]
