"""
Automatic discovery of butterfly‑compilable seed matrices.

Provides brute‑force enumeration of small integer matrices and filters
by solvability (Lie‑algebra derived series) and thermodynamic properties.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np

from .solvability import solvability_series
from .thermodynamics import SeedThermodynamics


def enumerate_solvable_seeds(
    entry_values: Iterable[int] = (-1, 0, 1),
    d: int = 2,
    tol: float = 1e-10,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """
    Enumerate all d×d integer matrices with entries in `entry_values`
    that are butterfly‑compilable (solvable Lie algebra).
    If `max_samples` is given, randomly sample matrices instead of
    exhaustive enumeration.

    Parameters
    ----------
    entry_values : iterable of int
        Allowed integer entries (default {-1,0,1}).
    d : int
        Seed dimension (default 2).
    tol : float
        Tolerance for solvability series.
    max_samples : int, optional
        If provided, sample at most this many random matrices.
        If None, enumerate all possible matrices (exponential in d²).
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    list of np.ndarray
        List of solvable seed matrices (each shape (d,d), dtype float64).
    """
    entry_list = list(entry_values)
    solvable = []

    rng = np.random.default_rng(random_state)

    if max_samples is None:
        # Exhaustive enumeration
        for flat in itertools.product(entry_list, repeat=d * d):
            S = np.array(flat, dtype=np.float64).reshape(d, d)
            result = solvability_series(S, tol=tol)
            if result["is_solvable"]:
                solvable.append(S)
    else:
        # Random sampling with replacement
        seen: set[tuple[int, ...]] = set()
        for _ in range(max_samples):
            flat = rng.choice(entry_list, size=d * d)
            key = tuple(int(v) for v in flat)
            if key in seen:
                continue
            seen.add(key)
            S = np.array(flat, dtype=np.float64).reshape(d, d)
            result = solvability_series(S, tol=tol)
            if result["is_solvable"]:
                solvable.append(S)

    return solvable


def enumerate_seeds_with_properties(
    entry_values: Iterable[int] = (-1, 0, 1),
    d: int = 2,
    *,
    unitary: bool | None = None,
    conservative: bool | None = None,
    contractive: bool | None = None,
    expansive: bool | None = None,
    nilpotent: bool | None = None,
    tol: float = 1e-10,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """
    Enumerate d×d integer matrices with entries in `entry_values`
    and filter by thermodynamic properties.
    If `max_samples` is given, randomly sample matrices instead of
    exhaustive enumeration.

    Parameters
    ----------
    entry_values : iterable of int
        Allowed integer entries (default {-1,0,1}).
    d : int
        Seed dimension (default 2).
    unitary, conservative, contractive, expansive, nilpotent : bool or None
        If not None, require the corresponding thermodynamic property.
    tol : float
        Tolerance for classification.
    max_samples : int, optional
        If provided, sample at most this many random matrices.
        If None, enumerate all possible matrices (exponential in d²).
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    list of np.ndarray
        List of seed matrices satisfying all requested properties.
    """
    entry_list = list(entry_values)
    matches = []

    rng = np.random.default_rng(random_state)

    if max_samples is None:
        # Exhaustive enumeration
        for flat in itertools.product(entry_list, repeat=d * d):
            S = np.array(flat, dtype=np.float64).reshape(d, d)
            thermo = SeedThermodynamics.analyze(S, tol=tol)

            ok = True
            if unitary is not None and thermo.is_unitary != unitary:
                ok = False
            if conservative is not None and thermo.is_conservative != conservative:
                ok = False
            if contractive is not None and thermo.is_contractive != contractive:
                ok = False
            if expansive is not None and thermo.is_expansive != expansive:
                ok = False
            if nilpotent is not None and thermo.is_nilpotent != nilpotent:
                ok = False
            if ok:
                matches.append(S)
    else:
        # Random sampling with replacement
        seen: set[tuple[int, ...]] = set()
        for _ in range(max_samples):
            flat = rng.choice(entry_list, size=d * d)
            key = tuple(int(v) for v in flat)
            if key in seen:
                continue
            seen.add(key)
            S = np.array(flat, dtype=np.float64).reshape(d, d)
            thermo = SeedThermodynamics.analyze(S, tol=tol)

            ok = True
            if unitary is not None and thermo.is_unitary != unitary:
                ok = False
            if conservative is not None and thermo.is_conservative != conservative:
                ok = False
            if contractive is not None and thermo.is_contractive != contractive:
                ok = False
            if expansive is not None and thermo.is_expansive != expansive:
                ok = False
            if nilpotent is not None and thermo.is_nilpotent != nilpotent:
                ok = False
            if ok:
                matches.append(S)

    return matches


def print_discovered_seeds(
    entry_values: Iterable[int] = (-1, 0, 1),
    d: int = 2,
    tol: float = 1e-10,
    max_samples: int | None = None,
    random_state: int | None = None,
) -> None:
    """
    Print all butterfly‑compilable seeds together with their thermodynamic
    classification.

    Parameters
    ----------
    entry_values, d, tol, max_samples, random_state
        As in `enumerate_solvable_seeds`.
    """
    seeds = enumerate_solvable_seeds(
        entry_values,
        d,
        tol,
        max_samples=max_samples,
        random_state=random_state,
    )
    print(
        f"Found {len(seeds)} solvable seeds (d={d}, entries in {list(entry_values)}):"
    )
    for idx, S in enumerate(seeds):
        thermo = SeedThermodynamics.analyze(S, tol=tol)
        print(f"\nSeed #{idx + 1}:")
        print(S)
        print(f"  unitary:      {thermo.is_unitary}")
        print(f"  conservative: {thermo.is_conservative}")
        print(f"  contractive:  {thermo.is_contractive}")
        print(f"  expansive:    {thermo.is_expansive}")
        print(f"  nilpotent:    {thermo.is_nilpotent}")
        print(f"  spectral radius: {thermo.spectral_radius:.6f}")
