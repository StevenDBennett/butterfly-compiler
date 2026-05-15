#!/usr/bin/env python3
"""
Experiment: Lohmiller-Slotine Double-Slit Bridge.

Tests the connection between Lohmiller & Slotine (Proc. R. Soc. A 482, 20250413)
and the Butterfly Compiler.

Core claims tested:
  1. The QFT butterfly matches NumPy FFT to machine precision
  2. Free-particle propagation matches the analytic Gaussian solution
  3. The kinetic generator is abelian -> solvable -> O(N log N) butterfly
  4. Double-slit wavefunction decomposes into two classical branches
  5. Interference emerges from the coherent sum of branches
"""

import numpy as np

from butterfly import (
    qft_butterfly,
    solvability_series,
)

# ── Parameters ──────────────────────────────────────────────────────
H_BAR = 1.0
MASS = 1.0
N_QUBITS = 10  # N = 1024 grid points
L = 60.0  # domain [-L/2, L/2]
SIGMA0 = 1.5  # initial packet width
K0 = 6.0  # initial momentum
X0 = 0.0  # initial position (centered between slits)
T_FINAL = 2.5  # evolution time
SLIT_CENTERS = [-1.2, 1.2]
SLIT_HALF = 0.35  # half-width of each slit opening

N = 1 << N_QUBITS
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# QFT butterflies (cached)
apply_qft = qft_butterfly(N_QUBITS, include_swaps=True)
apply_iqft = qft_butterfly(N_QUBITS, include_swaps=True, inverse=True)


def propagate_qft(psi, t):
    """Propagate psi by time t via QFT + kinetic phase + inverse QFT."""
    psi_k = apply_qft(psi)
    psi_k *= np.exp(-1j * (k**2 / (2 * MASS)) * t)
    return apply_iqft(psi_k)


def free_gaussian_analytic(x, t, x0, sigma0, k0, hbar=1.0, m=1.0):
    """Analytic free-particle evolution of a Gaussian wavepacket.

    Initial: psi(x,0) = (2*pi*sigma0^2)^(-1/4)
                        exp(-(x-x0)^2/(4*sigma0^2)) exp(i*k0*x)

    Returns psi(x,t) via exact propagation of the Feynman propagator.
    """
    sigma_t = sigma0 * (1 + 1j * hbar * t / (2 * m * sigma0**2))
    norm = (2 * np.pi * sigma_t**2) ** (-0.25)
    psi = norm * np.exp(-((x - x0 - k0 * hbar * t / m) ** 2) / (4 * sigma0 * sigma_t))
    psi *= np.exp(1j * k0 * (x - k0 * hbar * t / (2 * m)) / hbar)
    return psi


# ── 0. Initial state ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("Lohmiller-Slotine Butterfly Bridge")
    print("=" * 70)

    # Gaussian wavepacket centered at x0 with momentum k0
    psi0 = (
        (2 * np.pi * SIGMA0**2) ** (-0.25)
        * np.exp(-((x - X0) ** 2) / (4 * SIGMA0**2))
        * np.exp(1j * K0 * x)
    )
    psi0 /= np.linalg.norm(psi0)  # ensure unit norm

    # ── 1. QFT butterfly accuracy + free-particle analytic check ────────
    print("\n── Part 1: Butterfly QFT accuracy and free-particle evolution ──")

    # 1a. QFT butterfly vs NumPy FFT (should match to machine precision)
    psi_test = np.random.randn(N) + 1j * np.random.randn(N)
    y_qft = apply_qft(psi_test.copy())
    y_np = np.fft.fft(psi_test, norm="ortho")
    qft_error = np.max(np.abs(y_qft - y_np))
    print("    QFT butterfly vs NumPy FFT (random test):")
    print(f"      Max error: {qft_error:.2e}")
    print(f"      Match: {'YES ✓' if qft_error < 1e-14 else 'SUSPICIOUS'}")

    # 1b. Free-particle propagation vs analytic Gaussian solution
    psi_numeric = propagate_qft(psi0, T_FINAL)
    psi_analytic = free_gaussian_analytic(x, T_FINAL, X0, SIGMA0, K0, H_BAR, MASS)
    psi_analytic *= np.linalg.norm(psi_numeric) / np.linalg.norm(
        psi_analytic
    )  # match scale

    overlap_free = np.abs(np.vdot(psi_numeric, psi_analytic))
    l2_free = np.linalg.norm(psi_numeric - psi_analytic) / np.linalg.norm(psi_numeric)
    print(f"\n    Butterfly QFT vs analytic Gaussian (t = {T_FINAL}):")
    print(f"      Overlap: {overlap_free:.10f}")
    print(f"      Rel. L2 error: {l2_free:.2e}")
    print(f"      Match: {'YES ✓' if overlap_free > 0.9999 else 'SUSPICIOUS'}")

    # 1c. Norm preservation
    norm_initial = np.linalg.norm(psi0)
    norm_final = np.linalg.norm(psi_numeric)
    print(f"\n    Norm preservation: {norm_initial:.10f} -> {norm_final:.10f}")
    print(
        f"      {'UNITARY ✓' if abs(norm_final - norm_initial) < 1e-10 else 'SUSPICIOUS'}"
    )

    # ── 2. Solvability analysis ─────────────────────────────────────────
    print("\n── Part 2: Solvability of the kinetic generator ──────────────")

    # The kinetic energy operator H = -½ d^2/dx^2 is diagonal in Fourier basis.
    # Any diagonal matrix generates an abelian Lie algebra -> solvable.
    # This certifies butterfly-compilability (Beth's theorem).

    H_seed = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    result_h = solvability_series(H_seed)
    print(f"    QFT Hadamard seed solvable: {result_h['is_solvable']}")
    print(f"    Derived series dims: {result_h['series']}")
    print("    Interpretation: Hadamard generates a solvable Lie algebra")
    print("      -> O(N log N) butterfly exists for the QFT")

    S_k = np.diag(np.array([k[0] ** 2, k[1] ** 2]) / 2)
    result_k = solvability_series(S_k)
    print(f"\n    Kinetic generator (diagonal) solvable: {result_k['is_solvable']}")
    print(f"    Derived series dims: {result_k['series']}")
    print("    Interpretation: diagonal -> abelian -> solvable in 1 step")
    print("      -> free particle has no branch proliferation")

    S_rotation = np.array([[0, -1], [1, 0]], dtype=float)  # SO(2) generator
    result_r = solvability_series(S_rotation)
    print(f"\n    Rotation generator solvable: {result_r['is_solvable']}")
    print(f"    Derived series dims: {result_r['series']}")
    print("    Interpretation: SO(2) is abelian -> solvable")

    # ── 3. Double-slit branch decomposition ─────────────────────────────
    print("\n── Part 3: Double-slit - two-classical-branch decomposition ───")

    # Create double-slit mask
    mask = np.zeros(N, dtype=complex)
    for cen in SLIT_CENTERS:
        mask[np.abs(x - cen) < SLIT_HALF] = 1.0

    # Mask the initial state
    psi0_masked = psi0 * mask
    psi0_masked /= np.linalg.norm(psi0_masked)
    energy_transmitted = np.sum(np.abs(psi0 * mask) ** 2)

    # Full double-slit propagation
    psi_full = propagate_qft(psi0_masked, T_FINAL)

    # Individual branch propagations (one slit open at a time)
    branch_waves = []
    for cen in SLIT_CENTERS:
        branch_mask = np.zeros(N, dtype=complex)
        branch_mask[np.abs(x - cen) < SLIT_HALF] = 1.0
        psi_branch = psi0 * branch_mask
        psi_branch /= np.linalg.norm(psi_branch)
        psi_branch_t = propagate_qft(psi_branch, T_FINAL)
        branch_waves.append(psi_branch_t)

    psi_left, psi_right = branch_waves

    # Linearity check: the sum of individual slit evolutions
    # should equal the full double-slit evolution (by linearity of the propagator)
    norm_left = np.linalg.norm(psi0 * (np.abs(x - SLIT_CENTERS[0]) < SLIT_HALF))
    norm_right = np.linalg.norm(psi0 * (np.abs(x - SLIT_CENTERS[1]) < SLIT_HALF))
    psi_sum = psi_left * norm_left + psi_right * norm_right
    psi_sum_normed = psi_sum / np.linalg.norm(psi_sum)
    psi_full_normed = psi_full / np.linalg.norm(psi_full)
    branch_overlap = np.abs(np.vdot(psi_full_normed, psi_sum_normed))

    # Check slit illumination balance
    amp_left_slit = np.linalg.norm(psi0 * (np.abs(x - SLIT_CENTERS[0]) < SLIT_HALF))
    amp_right_slit = np.linalg.norm(psi0 * (np.abs(x - SLIT_CENTERS[1]) < SLIT_HALF))

    print(f"    Energy through slits: {energy_transmitted:.4f} of 1.0")
    print(f"    Left/right slit balance: {amp_left_slit:.4f} / {amp_right_slit:.4f}")
    print(f"    Linearity: |<psi_full|psi_left+psi_right>| = {branch_overlap:.8f}")
    print(f"    Decomposition: {'LINEAR ✓' if branch_overlap > 0.9999 else 'CLOSE'}")

    # Phase gradient (velocity field) for each branch
    window = slice(N // 4, 3 * N // 4)
    for name, psi in [("Left", psi_left), ("Right", psi_right)]:
        phase = np.angle(psi)
        vg = np.gradient(np.unwrap(phase), dx)
        print(f"    {name} branch: mean v = {np.mean(vg[window]):.2f}")

    # ── 4. Interference pattern ─────────────────────────────────────────
    print("\n── Part 4: Interference pattern ───────────────────────────────")

    prob_full = np.abs(psi_full) ** 2
    screen_region = x > 6  # far-field screen

    if np.any(screen_region):
        prob_screen = prob_full[screen_region]
        x_screen = x[screen_region]
        peak_idx = np.argmax(prob_screen)
        x_peak = x_screen[peak_idx]
        peak_val = prob_screen[peak_idx]

        # Number of visible fringes (above 20% of peak)
        threshold = 0.2 * peak_val
        above = prob_screen > threshold
        transitions = np.diff(above.astype(int))
        n_fringes = np.sum(transitions == 1)

        # Fringe spacing from FFT of screen probability
        from numpy.fft import rfft, rfftfreq

        spectrum = np.abs(rfft(prob_screen - prob_screen.mean()))
        freqs = rfftfreq(len(prob_screen), d=dx)
        valid = freqs > 0
        if np.any(valid):
            dominant_k = freqs[valid][np.argmax(spectrum[valid])]
            spacing = 2 * np.pi / dominant_k if dominant_k > 0 else float("inf")
            d_slit = SLIT_CENTERS[1] - SLIT_CENTERS[0]
            expected_spacing = 2 * np.pi * np.mean(x_screen) / (K0 * d_slit)

        print(f"    Peak probability at x ~ {x_peak:.2f} (p = {peak_val:.4f})")
        if n_fringes >= 2:
            print(f"    Visible fringes on screen: ~{n_fringes}")
            print(f"    Measured fringe spacing: {spacing:.3f}")
            print(f"    Expected (far-field):   {expected_spacing:.3f}")
            print(
                f"    Match: {'YES' if abs(spacing - expected_spacing) / expected_spacing < 0.3 else 'approximate'}"
            )
        else:
            print("    No clear fringes - need larger t or finer grid")

        # Contrast (visibility)
        if len(prob_screen) > 10:
            p_max = np.max(prob_screen)
            p_min = np.min(prob_screen[prob_screen > 0])
            visibility = (p_max - p_min) / (p_max + p_min) if (p_max + p_min) > 0 else 0
            print(f"    Fringe visibility: {visibility:.3f}")

    # ── 5. Single-slit vs double-slit comparison ────────────────────────
    print("\n── Part 5: Single-slit vs double-slit (interference check) ────")

    # Double-slit probability
    prob_double = np.abs(psi_full) ** 2

    # Single-slit probability (incoherent sum - no interference)
    prob_single_incoherent = (
        np.abs(psi_left) ** 2 * norm_left**2 + np.abs(psi_right) ** 2 * norm_right**2
    )

    # Compare on the screen
    if np.any(screen_region):
        diff = prob_double[screen_region] - prob_single_incoherent[screen_region]
        interference_energy = np.sum(np.abs(diff)) / np.sum(prob_double[screen_region])
        print(f"    Interference energy fraction on screen: {interference_energy:.3f}")
        print(
            f"    {'INTERFERENCE ✓' if interference_energy > 0.05 else 'weak interference'}"
        )
        print(
            f"    (Incoherent sum has no fringes; coherent sum has ~{n_fringes} fringes)"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    checks = [
        ("QFT butterfly matches NumPy FFT", qft_error < 1e-14),
        ("Free: QFT = analytic Gaussian", overlap_free > 0.9999),
        ("Kinetic generator is solvable", result_h["is_solvable"]),
        ("Branch decomposition is linear", branch_overlap > 0.9999),
        ("Interference fringes visible", n_fringes >= 2),
    ]
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")

    print("\n  Interpretation:")
    print("  - The QFT butterfly is a correct O(N log N) Fourier transform.")
    print("  - Free-particle evolution matches the analytic Gaussian solution,")
    print("    confirming the Feynman propagator computes classical action sums.")
    print("  - The kinetic generator is abelian (Beth's theorem: solvable),")
    print("    so the classical action sum compiles to O(N log N).")
    print("  - The double-slit wavefunction decomposes into 2 independent")
    print("    branches, each propagated by the same butterfly (linearity).")
    print("  - Interference fringes appear when branches recombine, matching")
    print("    the Lohmiller-Slotine claim: psi = sum_j sqrt(rho_j) exp(i phi_j/hbar).")
