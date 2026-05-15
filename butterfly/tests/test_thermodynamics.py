"""
Tests for thermodynamic seed classification.
"""

import numpy as np

from butterfly import DEFAULT_TOL, SeedThermodynamics, nilpotency_index


def run_tests_thermodynamics(check, rng):
    """Tests for thermodynamic classification."""
    print("\n── Thermodynamic Classification ───────────────────────────────────")
    h2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    thermo_h = SeedThermodynamics.analyze(h2)
    check(
        "Hadamard is unitary/conservative",
        thermo_h.is_unitary and thermo_h.is_conservative,
    )
    check(
        "Hadamard spectral radius = 1",
        abs(thermo_h.spectral_radius - 1.0) < DEFAULT_TOL,
    )

    Z = np.array([[1.0, 0.0], [1.0, 1.0]])
    thermo_z = SeedThermodynamics.analyze(Z)
    check("Zeta is NOT unitary", not thermo_z.is_unitary)
    check(
        "Zeta is conservative but not unitary",
        thermo_z.is_conservative and not thermo_z.is_unitary,
    )

    F = np.array([[1.0, 1.0], [1.0, 0.0]])
    thermo_f = SeedThermodynamics.analyze(F)
    check("Fibonacci is expansive", thermo_f.is_expansive)
    check("Fibonacci λ_max = φ", abs(thermo_f.spectral_radius - 1.618) < 0.01)

    Nmat = np.array([[0.0, 1.0], [0.0, 0.0]])
    t_N = SeedThermodynamics.analyze(Nmat)
    check("Nilpotent matrix detected", t_N.is_nilpotent)
    check("Non-nilpotent (Hadamard) not flagged", not thermo_h.is_nilpotent)

    C = np.array([[0.5, 0.0], [0.0, 0.5]])
    thermo_c = SeedThermodynamics.analyze(C)
    check(
        "Contractive matrix has spectral radius < 1",
        thermo_c.spectral_radius < 1.0 - DEFAULT_TOL,
    )
    check("Contractive matrix is not unitary", not thermo_c.is_unitary)
    check("Contractive matrix is not expansive", not thermo_c.is_expansive)
    check("Contractive matrix is not nilpotent", not thermo_c.is_nilpotent)
    check(
        "Contractive diagonal matrix is NOT conservative", not thermo_c.is_conservative
    )

    idx = nilpotency_index(Nmat)
    check("Nilpotency index of [[0,1],[0,0]] is 2", idx == 2)
    idx2 = nilpotency_index(h2)
    check("Non-nilpotent matrix returns None", idx2 is None)

    thermo_default = SeedThermodynamics.analyze(h2, tol=DEFAULT_TOL)
    thermo_small = SeedThermodynamics.analyze(h2, tol=1e-15)
    check(
        "Thermodynamic classification consistent with small tolerance",
        thermo_default.is_unitary == thermo_small.is_unitary,
    )
