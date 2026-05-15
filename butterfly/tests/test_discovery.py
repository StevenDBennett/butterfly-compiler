"""
Tests for seed discovery utilities.
"""

import io
import sys

import numpy as np

from butterfly import (
    SeedThermodynamics,
    enumerate_seeds_with_properties,
    enumerate_solvable_seeds,
    print_discovered_seeds,
    solvability_series,
)


def run_tests_discovery(check, rng):
    """Tests for seed discovery."""
    print("\n── Seed discovery ──────────────────────────────────────────────")
    solvable = enumerate_solvable_seeds(d=2, entry_values=(-1, 0, 1))
    check("At least one solvable seed found", len(solvable) > 0)
    for idx, S in enumerate(solvable):
        result = solvability_series(S)
        check(f"Seed #{idx + 1} is solvable", result["is_solvable"])

    unitary = enumerate_seeds_with_properties(
        d=2, entry_values=(-1, 0, 1), unitary=True
    )
    check("Unitary seeds found", len(unitary) > 0)
    for S in unitary:
        thermo = SeedThermodynamics.analyze(S)
        check("Unitary seed is indeed unitary", thermo.is_unitary)

    sampled = enumerate_solvable_seeds(
        d=2, entry_values=(-1, 0, 1), max_samples=10, random_state=42
    )
    check("Random sampling returns at most max_samples seeds", len(sampled) <= 10)

    sampled2 = enumerate_solvable_seeds(
        d=2, entry_values=(-1, 0, 1), max_samples=10, random_state=42
    )
    check(
        "Random sampling with same random_state is deterministic",
        len(sampled) == len(sampled2)
        and all(np.array_equal(a, b) for a, b in zip(sampled, sampled2, strict=True)),
    )

    unitary_sampled = enumerate_seeds_with_properties(
        d=2, entry_values=(-1, 0, 1), unitary=True, max_samples=20, random_state=123
    )
    check(
        "Property filtering with random sampling works",
        all(SeedThermodynamics.analyze(S).is_unitary for S in unitary_sampled),
    )

    captured = io.StringIO()
    sys.stdout = captured
    try:
        print_discovered_seeds(d=2, entry_values=(-1, 0, 1))
    finally:
        sys.stdout = sys.__stdout__
    check("print_discovered_seeds runs without error", True)
