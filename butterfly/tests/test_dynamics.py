"""
Tests for continuous-time fast-forwarding and Dyson series.
"""

import numpy as np

from butterfly.dynamics import (
    continuous_fast_forward,
    dyson_first_order,
    dyson_rank1_correction,
    separable_heat_solve,
)


def run_tests_dynamics(check, rng):
    """Tests for dynamics (continuous fast-forward and Dyson series)."""
    print("\n── Dynamics (Continuous Fast-Forward & Dyson Series) ─────────────")

    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    v0 = rng.standard_normal(4)
    t = 1.0

    vt = continuous_fast_forward(v0, A, t)
    check("continuous_fast_forward returns correct length", len(vt) == 4)

    Asmall = np.array([[0.0, -0.1], [0.1, 0.0]])
    v0small = np.array([1.0, 0.0])
    _ = continuous_fast_forward(v0small, Asmall, t)
    vt_direct = continuous_fast_forward(v0small, Asmall, 0.0)
    check(
        "continuous_fast_forward at t=0 returns input", np.allclose(v0small, vt_direct)
    )

    u0 = rng.standard_normal(4)
    L_1d = np.array([[-1.0, 0.5], [0.5, -1.0]])
    ut = separable_heat_solve(u0, L_1d, t=0.1)
    check("separable_heat_solve returns correct length", len(ut) == 4)

    A_2x2 = np.array([[-0.1, 0.0], [0.0, -0.2]], dtype=float)
    v0_2 = rng.standard_normal(4)
    t_2 = 0.5
    vt2 = continuous_fast_forward(v0_2, A_2x2, t_2)
    check("continuous_fast_forward with diagonal generator works", len(vt2) == 4)

    v0_3 = rng.standard_normal(4)
    A_3 = np.array([[0.0, -1.0], [1.0, 0.0]])
    B_dense = 0.1 * rng.standard_normal((4, 4))
    t_3 = 0.2
    v_dyson = None
    try:
        v_dyson = dyson_first_order(v0_3, A_3, B_dense, t_3, n_quadrature=4)
        check(
            "dyson_first_order with dense B returns correct length", len(v_dyson) == 4
        )
    except Exception as e:
        check(f"dyson_first_order runs without error: {e}", False)

    u_vec = rng.standard_normal(4)
    w_vec = rng.standard_normal(4)
    try:
        v_rank1 = dyson_rank1_correction(v0_3, A_3, u_vec, w_vec, t_3, n_quadrature=4)
        check("dyson_rank1_correction returns correct length", len(v_rank1) == 4)
    except Exception as e:
        check(f"dyson_rank1_correction runs without error: {e}", False)
