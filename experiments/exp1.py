#!/usr/bin/env python3
"""Test the Butterfly Compiler - classification + QFT double-slit demo."""

import numpy as np

# Assume butterfly is installed or butterfly_compiler.py is in the same directory
try:
    from butterfly import (
        REAL_SEMIRING,
        SeedThermodynamics,
        fast_kron_power_transform,
        qft_butterfly,
        solvability_series,
    )
except ImportError:
    print(
        "Error: butterfly module not found. Make sure butterfly_compiler.py is in your PYTHONPATH."
    )
    exit(1)

# ------------------------------------------------------------
# 1. Seed classification and solvability
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Seed analysis ===")
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # normalized Hadamard

    thermo = SeedThermodynamics.analyze(S)
    print(f"Hadamard seed:\n{S}")
    print(f"Unitary: {thermo.is_unitary}")
    print(f"Conservative: {thermo.is_conservative}")
    print(f"Spectral radius: {thermo.spectral_radius:.4f}")

    solv = solvability_series(S)
    print(f"Solvable (Beth's theorem): {solv['is_solvable']}")
    print(f"Derived series depth: {len(solv['series'])}")

    # ------------------------------------------------------------
    # 2. Fast Kronecker-power transform (real semiring)
    # ------------------------------------------------------------
    print("\n=== Fast transform ===")
    N = 8  # 2^3, so n=3
    x = np.random.randn(N)
    y = fast_kron_power_transform(x, S, REAL_SEMIRING)
    print(f"Input vector (size {N}): first 3 entries = {x[:3]}")
    print(f"Output after S^(⊗3): first 3 entries = {y[:3]}")

    # ------------------------------------------------------------
    # 3. QFT double-slit demo (using butterfly's own QFT, no numpy.fft)
    # ------------------------------------------------------------
    print("\n=== QFT double-slit evolution (butterfly-only) ===")
    n_qubits = 6  # N = 64
    N = 2**n_qubits
    L = 10.0
    x_pos = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x_pos[1] - x_pos[0]

    # initial Gaussian packet
    sigma = 0.4
    k0 = 4.0
    psi0 = np.exp(-((x_pos + 3) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x_pos)
    psi0 /= np.linalg.norm(psi0)

    # double slit (two openings)
    mask = np.zeros(N, dtype=complex)
    slit_cen = [-2.0, 2.0]
    slit_half = 0.3
    for cen in slit_cen:
        mask[np.abs(x_pos - cen) < slit_half] = 1.0
    psi0 *= mask

    # Get the butterfly QFT and inverse QFT (they are unitary)
    apply_qft = qft_butterfly(n_qubits, include_swaps=True)
    apply_iqft = qft_butterfly(n_qubits, include_swaps=True, inverse=True)

    # momentum grid for kinetic energy
    k = (
        2 * np.pi * np.fft.fftfreq(N, d=dx)
    )  # we still need np.fft for the grid, but not for transform

    # evolution time
    t = 1.5
    m = 1.0

    # kinetic propagator in momentum space
    print("Applying QFT (butterfly) -> phase -> inverse QFT")
    psi_k = apply_qft(psi0)  # forward QFT
    phase = np.exp(-1j * (k**2 / (2 * m)) * t)
    psi_k *= phase
    psi_t = apply_iqft(psi_k)  # inverse QFT

    # simple output: probability at final time
    prob = np.abs(psi_t) ** 2
    print(
        f"After t = {t:.2f}, max probability = {prob.max():.4f} at x ≈ {x_pos[np.argmax(prob)]:.2f}"
    )
    print("First few probability values:", prob[:5])

    # optional: verify unitarity (should be close to 1)
    print(
        f"Norm preservation: initial norm = {np.linalg.norm(psi0):.6f}, final norm = {np.linalg.norm(psi_t):.6f}"
    )
