"""
Tests for solvability analysis via Lie-algebra derived series.
"""

import numpy as np

from butterfly import DEFAULT_TOL, solvability_series


def run_tests_solvability(check, rng):
    """Tests for solvability analysis."""
    print("\n── Solvability ─────────────────────────────────────────────────────")
    Z = np.array([[1.0, 0.0], [1.0, 1.0]])
    h2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    sol_Z = solvability_series(Z)
    check("Zeta is SOLVABLE (triangular)", sol_Z["is_solvable"])
    sol_H = solvability_series(h2)
    check("Hadamard is solvable", sol_H["is_solvable"])

    R = np.array([[0, -1], [1, 0]], dtype=float)
    sol_R = solvability_series(R)
    check("90° rotation is solvable", sol_R["is_solvable"])

    Nmat = np.array([[0.0, 1.0], [0.0, 0.0]])
    Nsol = solvability_series(Nmat)
    check("Nilpotent matrix is solvable", Nsol["is_solvable"])

    sol_Z_adj = solvability_series(Z, include_adjoint=True)
    check(
        "⟨Z, Z†⟩_Lie is NOT solvable (= gl_2)",
        not sol_Z_adj["is_solvable"],
        f"series={sol_Z_adj.get('series')}",
    )

    for name, S in [
        ("Zeta", Z),
        ("Hadamard", h2),
        ("Rotation", R),
        ("Nilpotent", Nmat),
    ]:
        svd_res = solvability_series(S, method="svd")
        inc_res = solvability_series(S, method="incremental")
        check(
            f"{name} solvability consistent (svd vs incremental)",
            svd_res["is_solvable"] == inc_res["is_solvable"],
        )
        check(
            f"{name} series consistent (svd vs incremental)",
            svd_res["series"] == inc_res["series"],
        )

    S3 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    svd_res = solvability_series(S3, method="svd")
    inc_res = solvability_series(S3, method="incremental")
    check(
        "Random 3x3 solvability consistent (svd vs incremental)",
        svd_res["is_solvable"] == inc_res["is_solvable"],
    )
    check(
        "Random 3x3 series consistent (svd vs incremental)",
        svd_res["series"] == inc_res["series"],
    )

    for name, S in [
        ("Zeta", Z),
        ("Hadamard", h2),
        ("Rotation", R),
        ("Nilpotent", Nmat),
    ]:
        adj_res = solvability_series(S, method="adjoint")
        ce_res = solvability_series(S, method="chevalley")
        check(
            f"{name} solvability consistent (adjoint vs chevalley)",
            adj_res["is_solvable"] == ce_res["is_solvable"],
        )
        check(
            f"{name} series consistent (adjoint vs chevalley)",
            adj_res["series"] == ce_res["series"],
        )

    S4 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    adj_res = solvability_series(S4, method="adjoint")
    ce_res = solvability_series(S4, method="chevalley")
    check(
        "Random 4x4 solvability consistent (adjoint vs chevalley)",
        adj_res["is_solvable"] == ce_res["is_solvable"],
    )
    check(
        "Random 4x4 series consistent (adjoint vs chevalley)",
        adj_res["series"] == ce_res["series"],
    )

    T = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    sol_T = solvability_series(T)
    check("Strictly upper triangular matrix is solvable", sol_T["is_solvable"])

    T_small = np.array([[1.0, 1e-10], [0.0, 1.0]])
    sol_default = solvability_series(T_small, tol=DEFAULT_TOL)
    sol_small = solvability_series(T_small, tol=1e-15)
    sol_large = solvability_series(T_small, tol=1e-5)
    check(
        "Solvability result consistent (default vs small)",
        sol_default["is_solvable"] == sol_small["is_solvable"],
    )
    check(
        "Solvability result consistent (default vs large)",
        sol_default["is_solvable"] == sol_large["is_solvable"],
    )

    try:
        _ = solvability_series(T_small, tol=0.0)
        check("solvability_series with tol=0.0 does not crash", True)
    except Exception as e:
        check(
            "solvability_series with tol=0.0 does not crash",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )

    try:
        _ = solvability_series(T_small, tol=-1e-10)
        check("solvability_series with negative tolerance does not crash", True)
    except Exception as e:
        check(
            "solvability_series with negative tolerance does not crash",
            False,
            detail=f"got {type(e).__name__}: {e}",
        )
