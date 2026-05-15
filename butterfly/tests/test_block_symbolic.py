"""
Tests for block butterfly and symbolic fast-forwarding.
"""

import numpy as np

from butterfly import (
    DEFAULT_TOL,
    BlockSeed,
    SymbolicMatrix,
    block_butterfly_transform,
    symbolic_kronecker_power,
    symbolic_matrix_power,
)


def run_tests_block_symbolic(check, rng):
    """Tests for block butterfly and symbolic operations."""
    print("\n── Block Butterfly / Hierarchical ────────────────────────────────")
    block_seed = BlockSeed(
        [[np.eye(2), np.zeros((2, 2))], [np.ones((2, 2)), np.eye(2)]]
    )
    check("Block seed is butterfly-compilable", block_seed.is_butterfly_compilable())

    blocks_id = [[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]]
    block_seed_id = BlockSeed(blocks_id)
    check(
        "BlockSeed with all identity blocks is butterfly-compilable",
        block_seed_id.is_butterfly_compilable(),
    )

    d = block_seed.d
    k = block_seed.block_size

    def ring_kronecker_power_matrix(blocks, m):
        if m == 1:
            d = len(blocks)
            k = blocks[0][0].shape[0]
            mat = np.zeros((d * k, d * k))
            for i in range(d):
                for j in range(d):
                    mat[i * k : (i + 1) * k, j * k : (j + 1) * k] = blocks[i][j]
            return mat
        else:
            small = ring_kronecker_power_matrix(blocks, m - 1)
            d = len(blocks)
            k = blocks[0][0].shape[0]
            d_small = small.shape[0] // k
            d_out = d * d_small
            result = np.zeros((d_out * k, d_out * k))
            for i1 in range(d):
                for j1 in range(d):
                    block_ij = blocks[i1][j1]
                    for i2 in range(d_small):
                        for j2 in range(d_small):
                            row = (i1 * d_small + i2) * k
                            col = (j1 * d_small + j2) * k
                            result[row : row + k, col : col + k] = (
                                block_ij
                                @ small[i2 * k : (i2 + 1) * k, j2 * k : (j2 + 1) * k]
                            )
            return result

    M1 = ring_kronecker_power_matrix(block_seed.blocks, 1)
    x1 = rng.standard_normal(d * k)
    y1_block = block_butterfly_transform(x1, block_seed)
    y1_explicit = M1 @ x1
    check(
        "Block butterfly m=1 matches explicit matrix",
        np.allclose(y1_block, y1_explicit, atol=DEFAULT_TOL),
    )

    m = 2
    N = k * (d**m)
    x2 = rng.standard_normal(N)
    y2_block = block_butterfly_transform(x2, block_seed)
    M2 = ring_kronecker_power_matrix(block_seed.blocks, m)
    y2_explicit = M2 @ x2
    check(
        "Block butterfly m=2 matches explicit ring Kronecker power",
        np.allclose(y2_block, y2_explicit, atol=DEFAULT_TOL),
    )

    print("\n── Symbolic Fast-Forwarding ───────────────────────────────────────")
    S_sym = SymbolicMatrix(
        {
            (0, 0, (0, 0)): 1.0,
            (0, 1, (1, 0)): 1.0,
            (1, 0, (0, 1)): 1.0,
        },
        max_degree=10,
        shape=(2, 2),
    )

    result_kron = symbolic_kronecker_power(
        S_sym, 3, row_tuple=(0, 0, 0), col_tuple=(0, 1, 0)
    )
    check(
        "True Kronecker entry (0,0,0),(0,1,0) = x",
        abs(result_kron.get((1, 0), 0.0) - 1.0) < 1e-12 and len(result_kron) == 1,
    )

    result_kron_id = symbolic_kronecker_power(
        S_sym, 3, row_tuple=(0, 0, 0), col_tuple=(0, 0, 0)
    )
    check(
        "True Kronecker entry (0,0,0),(0,0,0) = 1",
        abs(result_kron_id.get((0, 0), 0.0) - 1.0) < 1e-12 and len(result_kron_id) == 1,
    )

    result_mp = symbolic_matrix_power(S_sym, 3, entry=(0, 0))
    check(
        "Path polynomial (S^3)_{0,0} constant term = 1",
        abs(result_mp.get((0, 0), 0.0) - 1.0) < 1e-12,
    )
    check(
        "Path polynomial (S^3)_{0,0} xy coefficient = 2",
        abs(result_mp.get((1, 1), 0.0) - 2.0) < 1e-12,
    )
    check("Path polynomial (S^3)_{0,0} has exactly 2 monomials", len(result_mp) == 2)

    S1 = SymbolicMatrix.from_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]), n_vars=1)
    S2 = SymbolicMatrix.from_matrix(np.array([[0.5, 0.0], [1.5, 2.5]]), n_vars=1)
    S_sum = S1 + S2
    expected_sum = np.array([[1.5, 2.0], [1.5, 5.5]])
    S_sum_expected = SymbolicMatrix.from_matrix(expected_sum, n_vars=1)
    check(
        "SymbolicMatrix.__add__ works correctly", S_sum.coeffs == S_sum_expected.coeffs
    )

    A = SymbolicMatrix.from_matrix(np.array([[1, 2], [3, 4]]), n_vars=1)
    B_sym = SymbolicMatrix.from_matrix(np.array([[5, 6], [7, 8]]), n_vars=1)
    C = A @ B_sym
    expected_product = np.array(
        [[1 * 5 + 2 * 7, 1 * 6 + 2 * 8], [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]]
    )
    C_expected = SymbolicMatrix.from_matrix(expected_product, n_vars=1)
    check("SymbolicMatrix.__matmul__ works correctly", C.coeffs == C_expected.coeffs)
