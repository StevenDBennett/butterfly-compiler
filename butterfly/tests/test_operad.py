"""
Tests for operadic composition and compiled butterfly.
"""

import numpy as np

from butterfly import (
    GF2_SEMIRING,
    REAL_SEMIRING,
    ButterflyOperad,
    CompiledButterfly,
    MultiSeedOperad,
    fast_kron_power_transform,
)


def run_tests_operad(check, rng):
    """Tests for operadic composition."""
    print("\n── Operadic Composition ────────────────────────────────────────────")
    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    op1 = ButterflyOperad(H2)
    x_test = rng.standard_normal(8)
    y_op = op1.apply(x_test)
    y_direct = fast_kron_power_transform(x_test, op1.seed)
    check("Operad apply matches direct", np.allclose(y_op, y_direct))

    seedI = np.array([[1, 0], [0, 1]], dtype=float)
    seedSwap = np.array([[0, 1], [1, 0]], dtype=float)
    opI = ButterflyOperad(seedI)
    opSwap = ButterflyOperad(seedSwap)
    opComposed = opI.compose(opSwap)
    check(
        "ButterflyOperad.compose() returns MultiSeedOperad",
        isinstance(opComposed, MultiSeedOperad),
    )
    check(
        "Composed operad seeds match [I, swap]",
        len(opComposed.seeds) == 2
        and np.array_equal(opComposed.seeds[0], seedI)
        and np.array_equal(opComposed.seeds[1], seedSwap),
    )
    xc = np.array([1.0, 2.0, 3.0, 4.0])
    yc = opComposed.apply(xc)
    check(
        "Composed operad (I then swap) gives [3,4,1,2]",
        np.allclose(yc, [3.0, 4.0, 1.0, 2.0]),
    )

    seeds_ms = [seedI, seedSwap]
    ms_op = MultiSeedOperad(seeds_ms)
    x_ms = np.array([1.0, 2.0, 3.0, 4.0])
    y_ms = ms_op.apply(x_ms)
    check("Multi-seed (I then swap) gives [3,4,1,2]", np.allclose(y_ms, [3, 4, 1, 2]))

    seed_tw = np.array([[1, 0], [0, 1]], dtype=complex)
    twiddle = np.exp(1j * np.pi / 4 * np.arange(4))
    ms_tw = MultiSeedOperad([seed_tw], twiddles=[twiddle])
    x_tw = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    y_tw = ms_tw.apply(x_tw)
    check(
        "MultiSeedOperad with twiddle factors applies element-wise multiplication",
        np.allclose(y_tw, x_tw * twiddle),
    )

    print("\n── Compiled Butterfly & Nilpotency ──────────────────────────────")
    cb = CompiledButterfly(H2, REAL_SEMIRING)
    x = rng.standard_normal(8)
    y_cb = cb.apply(x)
    y_direct = fast_kron_power_transform(x, H2, REAL_SEMIRING)
    check("CompiledButterfly matches direct transform", np.allclose(y_cb, y_direct))

    zeta_gf2 = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    cb_gf2 = CompiledButterfly(zeta_gf2, GF2_SEMIRING)
    x_gf2 = rng.integers(0, 2, 16, dtype=np.uint8)
    y_cb_gf2 = cb_gf2.apply(x_gf2)
    y_direct_gf2 = fast_kron_power_transform(x_gf2, zeta_gf2, GF2_SEMIRING)
    check(
        "CompiledButterfly GF2 matches direct transform",
        np.array_equal(y_cb_gf2, y_direct_gf2),
    )
