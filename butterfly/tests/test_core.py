"""
Tests for core butterfly transforms and semiring variants.
"""

import numpy as np

from butterfly import (
    DEFAULT_TOL,
    GF2_SEMIRING,
    LOG_SEMIRING,
    REAL_SEMIRING,
    TROPICAL_MAX,
    TROPICAL_MIN,
    fast_forward,
    fast_gf2_transform,
    fast_kron_power_transform,
    fast_tropical_transform,
)


def run_tests_core(check, rng):
    """Tests for core transform and semiring variants."""
    print("\n── Semiring Variants ──────────────────────────────────────────────")
    x = rng.standard_normal(16)
    y_real = fast_kron_power_transform(
        x, np.array([[0.5, 0.5], [0.5, -0.5]]), REAL_SEMIRING
    )
    check("Real semiring works", len(y_real) == 16)

    d0 = np.full(16, np.inf)
    d0[0] = 0.0
    hamming = fast_kron_power_transform(
        d0, np.array([[0.0, 1.0], [1.0, 0.0]]), TROPICAL_MIN
    )
    expected = [bin(i).count("1") for i in range(16)]
    check("Tropical gives Hamming weights", np.array_equal(hamming, expected))

    inf = np.inf
    seed_max_id = np.array([[0.0, -inf], [-inf, 0.0]])
    x_max = np.array(
        [
            2.0,
            -inf,
            -inf,
            5.0,
            -inf,
            3.0,
            -inf,
            -inf,
            1.0,
            -inf,
            -inf,
            -inf,
            -inf,
            -inf,
            -inf,
            -inf,
        ]
    )
    y_max = fast_kron_power_transform(x_max, seed_max_id, TROPICAL_MAX)
    check("TROPICAL_MAX with identity seed returns input", np.array_equal(y_max, x_max))

    zeta_gf2 = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    delta_last = np.zeros(16, dtype=np.uint8)
    delta_last[15] = 1
    result_last = fast_kron_power_transform(delta_last, zeta_gf2, GF2_SEMIRING)
    check(
        "GF(2) Zeta^{⊗4}: delta_15 → all-ones",
        np.all(result_last == 1),
    )
    delta_first = np.zeros(16, dtype=np.uint8)
    delta_first[0] = 1
    result_first = fast_kron_power_transform(delta_first, zeta_gf2, GF2_SEMIRING)
    check(
        "GF(2) Zeta^{⊗4}: delta_0 → fixed point",
        np.array_equal(result_first, delta_first),
    )

    seed_log = np.array([[1.0, 0.5], [0.5, 1.0]])
    x_real = np.array([2.0, 3.0])
    y_real = fast_kron_power_transform(x_real, seed_log, REAL_SEMIRING)
    seed_log_domain = np.log(seed_log)
    x_log_domain = np.log(x_real)
    y_log = fast_kron_power_transform(x_log_domain, seed_log_domain, LOG_SEMIRING)
    check(
        "LOG_SEMIRING matches REAL_SEMIRING after exp",
        np.allclose(np.exp(y_log), y_real, atol=DEFAULT_TOL),
    )

    h2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    x_t = rng.standard_normal(8)
    y_t = fast_tropical_transform(x_t, h2, op="min-plus")
    check("fast_tropical_transform runs", y_t.shape == x_t.shape)
    y_gf2 = fast_gf2_transform(np.array([1, 0, 0, 0], dtype=np.uint8), zeta_gf2)
    check("fast_gf2_transform runs", y_gf2.shape == (4,))

    seed_2x2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    x_vec = np.array([1.0, 2.0])
    result_matvec = REAL_SEMIRING.matvec(seed_2x2, x_vec)
    result_direct = seed_2x2 @ x_vec
    check(
        "Semiring.matvec matches direct matrix multiplication for N=d",
        np.allclose(result_matvec, result_direct, atol=DEFAULT_TOL),
    )
    x_vec2 = np.array([1.0, 2.0, 3.0, 4.0])
    result_matvec2 = REAL_SEMIRING.matvec(seed_2x2, x_vec2)
    blocks = [x_vec2[:2], x_vec2[2:]]
    expected = np.zeros_like(x_vec2)
    for i in range(2):
        for j in range(2):
            expected[i * 2 : (i + 1) * 2] += seed_2x2[i, j] * blocks[j]
    check(
        "Semiring.matvec works for h>1",
        np.allclose(result_matvec2, expected, atol=DEFAULT_TOL),
    )

    print("\n── d > 2 seeds and edge cases ────────────────────────────────")
    for d_seed in (3, 4, 5):
        seed = rng.standard_normal((d_seed, d_seed))
        N = d_seed**2
        x = rng.standard_normal(N)
        y = fast_kron_power_transform(x, seed, REAL_SEMIRING)
        M_explicit = np.kron(seed, seed)
        y_explicit = M_explicit @ x
        check(
            f"d={d_seed} seed Kronecker power matches explicit",
            np.allclose(y, y_explicit, atol=DEFAULT_TOL),
        )

    seed2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    x1 = np.array([42.0])
    y1 = fast_kron_power_transform(x1, seed2, REAL_SEMIRING)
    check("N=1 returns unchanged vector", np.allclose(y1, x1))

    try:
        fast_kron_power_transform(np.array([]), seed2, REAL_SEMIRING)
        check("Empty input raises ValueError", False)
    except ValueError:
        check("Empty input raises ValueError", True)
    except Exception as e:
        check(
            "Empty input raises ValueError",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    try:
        fast_kron_power_transform(
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            REAL_SEMIRING,
        )
        check("Non-square seed raises ValueError", False)
    except ValueError:
        check("Non-square seed raises ValueError", True)
    except Exception as e:
        check(
            "Non-square seed raises ValueError",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    try:
        fast_kron_power_transform(
            np.array([1.0, 2.0, 3.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), REAL_SEMIRING
        )
        check("N not power of d raises ValueError", False)
    except ValueError:
        check("N not power of d raises ValueError", True)
    except Exception as e:
        check(
            "N not power of d raises ValueError",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    try:
        fast_kron_power_transform(
            np.array([1.0, 2.0]),
            np.array([[1, 1], [0, 1]], dtype=np.uint8),
            GF2_SEMIRING,
        )
        check("GF2 semiring with float input raises TypeError", False)
    except TypeError:
        check("GF2 semiring with float input raises TypeError", True)
    except Exception as e:
        check(
            "GF2 semiring with float input raises TypeError",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    try:
        fast_kron_power_transform(
            np.array([1, 0], dtype=np.uint8),
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            GF2_SEMIRING,
        )
        check("GF2 semiring with float seed raises TypeError", False)
    except TypeError:
        check("GF2 semiring with float seed raises TypeError", True)
    except Exception as e:
        check(
            "GF2 semiring with float seed raises TypeError",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    seed_complex = np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=complex)
    x_complex = np.array([1.0, 2.0, 3.0, 4.0])
    y_complex = fast_kron_power_transform(x_complex, seed_complex, REAL_SEMIRING)
    check("Complex seed with real semiring works", y_complex.shape[0] == 4)

    S_ff = np.array([[0.9, 0.1], [-0.1, 0.8]])
    v0 = rng.standard_normal(8)
    v_fast = fast_forward(v0, S_ff, t=5)
    v_iter = v0.copy()
    for _ in range(5):
        v_iter = fast_kron_power_transform(v_iter, S_ff)
    check(
        "fast_forward matches iteration", np.allclose(v_iter, v_fast, atol=DEFAULT_TOL)
    )
