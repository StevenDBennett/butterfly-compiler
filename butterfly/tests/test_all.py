"""
Comprehensive test suite for the Butterfly Compiler.

Aggregates all module-specific test files into a single runner.
Import this module for backward compatibility with butterfly_compiler.py.
"""

import sys
import time

import numpy as np

from butterfly import REAL_SEMIRING
from butterfly.tests.test_block_symbolic import run_tests_block_symbolic
from butterfly.tests.test_core import run_tests_core
from butterfly.tests.test_differentiable import run_tests_differentiable
from butterfly.tests.test_discovery import run_tests_discovery
from butterfly.tests.test_dynamics import run_tests_dynamics
from butterfly.tests.test_export import run_tests_export
from butterfly.tests.test_operad import run_tests_operad
from butterfly.tests.test_qft_ntt import run_tests_qft_ntt
from butterfly.tests.test_solvability import run_tests_solvability
from butterfly.tests.test_spectral import run_tests_spectral
from butterfly.tests.test_thermodynamics import run_tests_thermodynamics


def run_tests():
    """Run all tests by delegating to module-specific test runners."""
    rng = np.random.default_rng(42)
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  \u2713 {name}")
            passed += 1
        else:
            print(f"  \u2717 {name} {detail}")
            failed += 1

    print("=" * 70)
    print("  Butterfly Compiler v1.0 \\u2014 Test Suite")
    print("=" * 70)

    run_tests_thermodynamics(check, rng)
    run_tests_solvability(check, rng)
    run_tests_core(check, rng)
    run_tests_block_symbolic(check, rng)
    run_tests_operad(check, rng)
    run_tests_qft_ntt(check, rng)
    run_tests_export(check, rng)
    run_tests_differentiable(check, rng)
    run_tests_dynamics(check, rng)
    run_tests_spectral(check, rng)
    run_tests_discovery(check, rng)

    print(f"\n{'=' * 70}")
    print(f"  {passed} passed | {failed} failed")
    print(f"{'=' * 70}")
    return passed, failed


def run_benchmarks():
    """Performance benchmarks for key operations."""
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("  Butterfly Compiler v1.0 \\u2014 Performance Benchmarks")
    print("=" * 70)

    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    print(
        "\n\u2500\u2500 fast_kron_power_transform scaling (Hadamard seed, real semiring) \u2500\u2500"
    )
    for k in range(1, 13):
        n = 1 << k
        x = rng.standard_normal(n)
        start = time.perf_counter()
        from butterfly import fast_kron_power_transform

        _y = fast_kron_power_transform(x, H2, REAL_SEMIRING)
        elapsed = time.perf_counter() - start
        print(f"  n={n:5d} ({k:2d} bits): {elapsed * 1000:6.2f} ms")

    print(
        "\n\u2500\u2500 QFT butterfly vs numpy.fft \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    for n_q in (1, 2, 3, 4, 5, 6):
        size = 1 << n_q
        x = rng.standard_normal(size) + 1j * rng.standard_normal(size)

        from butterfly import qft_butterfly

        qft_apply = qft_butterfly(n_q, include_swaps=True)
        start = time.perf_counter()
        y_qft = qft_apply(x)
        t_qft = time.perf_counter() - start

        start = time.perf_counter()
        y_ref = np.fft.fft(x, norm="ortho")
        t_fft = time.perf_counter() - start

        err = float(np.max(np.abs(y_qft - y_ref)))
        print(
            f"  n={n_q}, size={size:4d}: QFT {t_qft * 1000:6.2f} ms, numpy {t_fft * 1000:6.2f} ms, err={err:.2e}"
        )

    print(
        "\n\u2500\u2500 Semiring performance comparison (size 256) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    size = 256
    x_real = rng.standard_normal(size)
    x_tropical = np.full(size, np.inf)
    x_tropical[0] = 0.0
    x_gf2 = np.zeros(size, dtype=np.uint8)
    x_gf2[0] = 1

    seeds = [
        ("Hadamard", H2, None),
        (
            "Zeta",
            np.array([[1, 1], [0, 1]], dtype=float),
            np.array([[1, 1], [0, 1]], dtype=np.uint8),
        ),
        (
            "Swap",
            np.array([[0, 1], [1, 0]], dtype=float),
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
        ),
    ]

    from butterfly import GF2_SEMIRING, TROPICAL_MIN, fast_kron_power_transform

    for name, seed_real, seed_gf2 in seeds:
        start = time.perf_counter()
        _ = fast_kron_power_transform(x_real, seed_real, REAL_SEMIRING)
        t_real = time.perf_counter() - start

        start = time.perf_counter()
        _ = fast_kron_power_transform(x_tropical, seed_real, TROPICAL_MIN)
        t_trop = time.perf_counter() - start

        if seed_gf2 is not None:
            start = time.perf_counter()
            _ = fast_kron_power_transform(x_gf2, seed_gf2, GF2_SEMIRING)
            t_gf2 = time.perf_counter() - start
            gf2_str = f", GF(2) {t_gf2 * 1000:5.2f} ms"
        else:
            gf2_str = ""

        print(
            f"  {name:8s}: real {t_real * 1000:5.2f} ms, tropical {t_trop * 1000:5.2f} ms{gf2_str}"
        )

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        run_benchmarks()
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_tests()
        print("\n" + "=" * 70)
        run_benchmarks()
    else:
        run_tests()
