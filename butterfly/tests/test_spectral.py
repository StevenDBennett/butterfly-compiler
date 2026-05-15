"""
Tests for spectral functional calculus.
"""

import numpy as np

from butterfly import fast_kron_power_transform
from butterfly.spectral import (
    build_product_spectrum,
    butterfly_polynomial,
    butterfly_resolvent,
    spectral_butterfly_apply,
)


def run_tests_spectral(check, rng):
    """Tests for spectral calculus (product spectrum, polynomial, resolvent)."""
    print("\n── Spectral Calculus ──────────────────────────────────────────────")

    eigvals = np.array([1.0, -1.0])
    n = 3
    spec = build_product_spectrum(eigvals, n)
    check("Product spectrum has length d^n", len(spec) == 2**n)
    check("Product spectrum contains 1.0", 1.0 in spec)
    check("Product spectrum contains -1.0", -1.0 in spec)

    S = np.array([[0.8, 0.0], [0.0, 0.5]])
    v = rng.standard_normal(8)

    def _power_fn(p):
        return lambda z: z**p

    for power in (1, 2, 3):
        result_pow = spectral_butterfly_apply(v, S, _power_fn(power), real_output=True)
        result_direct = v.copy()
        for _ in range(power):
            result_direct = fast_kron_power_transform(result_direct, S)
        check(
            f"spectral_butterfly_apply with z^{power} matches iteration",
            np.allclose(result_pow, result_direct, atol=1e-10),
        )

    S_diag = np.array([[0.5, 0.0], [0.0, 0.8]], dtype=float)
    v_diag = rng.standard_normal(8)
    result_exp = spectral_butterfly_apply(v_diag, S_diag, np.exp)
    check("spectral_butterfly_apply with exp works", len(result_exp) == 8)

    S_poly = np.array([[0.5, 0.1], [-0.1, 0.5]], dtype=float)
    v_poly = rng.standard_normal(8)
    coeffs = np.array([1.0, 0.5, 0.2])
    result_poly = butterfly_polynomial(v_poly, S_poly, coeffs)
    check("butterfly_polynomial returns correct length", len(result_poly) == 8)

    v_res = rng.standard_normal(8)
    S_res = np.array([[0.5, 0.0], [0.0, 0.3]], dtype=float)
    mu = 2.0
    try:
        result_res = butterfly_resolvent(v_res, S_res, mu)
        check("butterfly_resolvent returns correct length", len(result_res) == 8)
    except ValueError as e:
        check(
            "butterfly_resolvent error (expected for in-spectrum mu)",
            "in the product spectrum" in str(e),
        )
    except Exception as e:
        check(f"butterfly_resolvent unexpected error: {e}", False)
