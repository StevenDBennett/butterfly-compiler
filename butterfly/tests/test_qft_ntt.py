"""
Tests for QFT and NTT transforms.
"""

import numpy as np

from butterfly import (
    DEFAULT_TOL,
    MOD_NTT_998244353,
    Semiring,
    modular_semiring,
    naive_ntt,
    ntt_butterfly,
    qft_butterfly,
)
from butterfly.fwht import _gray_code_indices, fwht, sequency_permutation, xor_convolve
from butterfly.qft import _bit_reverse_indices


def run_tests_qft_ntt(check, rng):
    """Tests for QFT and NTT."""
    print("\n── QFT via Van Loan Factorization ────────────────────────────────")
    for n_q in (1, 2, 3, 4):
        qft_apply = qft_butterfly(n_q, include_swaps=True)
        x_q = rng.standard_normal(1 << n_q) + 1j * rng.standard_normal(1 << n_q)
        y_qft = qft_apply(x_q)
        y_ref = np.fft.fft(x_q, norm="ortho")
        err = float(np.max(np.abs(y_qft - y_ref)))
        check(
            f"QFT(n={n_q}) matches numpy.fft (max err={err:.2e})",
            np.allclose(y_qft, y_ref, atol=DEFAULT_TOL),
        )

    for n_q in (1, 2, 3, 4):
        qft_apply_no_swap = qft_butterfly(n_q, include_swaps=False)
        x_q = rng.standard_normal(1 << n_q) + 1j * rng.standard_normal(1 << n_q)
        y_no_swap = qft_apply_no_swap(x_q)
        br = _bit_reverse_indices(n_q)
        y_ref = np.fft.fft(x_q[br], norm="ortho")
        check(
            f"QFT(n={n_q}, include_swaps=False) matches DFT(bit_reverse(x))",
            np.allclose(y_no_swap, y_ref, atol=DEFAULT_TOL),
        )

    br3 = _bit_reverse_indices(3)
    check(
        "Bit-reversal for n=3 is correct", np.array_equal(br3, [0, 4, 2, 6, 1, 5, 3, 7])
    )

    print("\n── NTT (Number Theoretic Transform) ──────────────────────────────")
    p = 998244353
    g = 3
    n_ntt = 3
    N_ntt = 1 << n_ntt
    omega = pow(g, (p - 1) // N_ntt, p)
    assert pow(omega, N_ntt, p) == 1

    _, ntt_apply = ntt_butterfly(n_ntt, p, omega, inverse=False, include_swaps=True)
    x_ntt = rng.integers(0, p, N_ntt, dtype=np.int64)
    y_ntt = ntt_apply(x_ntt)
    y_naive = naive_ntt(x_ntt, omega, p)
    match = np.array_equal(y_ntt % p, y_naive % p)
    detail = ""
    if not match:
        diff_idx = np.where(y_ntt % p != y_naive % p)[0]
        detail = f" mismatches at indices {diff_idx[:5]}"
        for i in diff_idx[:3]:
            detail += f"\n    y_ntt[{i}]={y_ntt[i] % p}, y_naive[{i}]={y_naive[i] % p}"
    check("NTT matches naive computation", match, detail)

    alpha = rng.integers(1, p)
    beta = rng.integers(1, p)
    x1 = rng.integers(0, p, N_ntt, dtype=np.int64)
    x2 = rng.integers(0, p, N_ntt, dtype=np.int64)
    lhs = ntt_apply((alpha * x1 + beta * x2) % p)
    rhs = (alpha * ntt_apply(x1) + beta * ntt_apply(x2)) % p
    check("NTT linearity over ℤ/p", np.array_equal(lhs, rhs))

    _, intt_apply = ntt_butterfly(
        n_ntt, p, omega, inverse=True, include_swaps=True, scale=True
    )
    x_recovered = intt_apply(y_ntt)
    check(
        "Forward + inverse NTT recovers original (with scaling)",
        np.array_equal(x_ntt % p, x_recovered % p),
    )

    _, intt_no_scale = ntt_butterfly(
        n_ntt, p, omega, inverse=True, include_swaps=True, scale=False
    )
    y_scaled = intt_no_scale(y_ntt)
    expected = (x_ntt * N_ntt) % p
    check(
        "Inverse NTT without scaling multiplies by N",
        np.array_equal(y_scaled % p, expected % p),
    )

    semiring_mod = modular_semiring(p)
    seed_mod = np.array([[1, 2], [3, 4]], dtype=np.int64) % p
    x_mod = rng.integers(0, p, 2, dtype=np.int64)
    y_semiring = semiring_mod.matvec(seed_mod, x_mod)
    y_direct = (seed_mod @ x_mod) % p
    check(
        "Modular semiring matvec matches explicit multiplication",
        np.array_equal(y_semiring, y_direct),
    )

    check("MOD_NTT_998244353 is not None", MOD_NTT_998244353 is not None)
    check(
        "MOD_NTT_998244353 is a Semiring instance",
        isinstance(MOD_NTT_998244353, Semiring),
    )
    a = 123456789
    b = 987654321
    p = 998244353
    check(
        f"MOD_NTT_998244353 addition {a}+{b} mod {p}",
        MOD_NTT_998244353.add(a, b) == (a + b) % p,
    )
    check(
        f"MOD_NTT_998244353 multiplication {a}*{b} mod {p}",
        MOD_NTT_998244353.mul(a, b) == (a * b) % p,
    )
    seed_mod2 = np.array([[1, 2], [3, 4]], dtype=np.int64) % p
    x_mod2 = np.array([5, 6], dtype=np.int64) % p
    y_mod2 = MOD_NTT_998244353.matvec(seed_mod2, x_mod2)
    y_direct2 = (seed_mod2 @ x_mod2) % p
    check(
        "MOD_NTT_998244353.matvec matches explicit multiplication",
        np.array_equal(y_mod2, y_direct2),
    )

    print("\n── Fast Walsh‑Hadamard Transform ────────────────────────────────")
    n_fwht = 4
    N_fwht = 1 << n_fwht
    x_fwht = rng.standard_normal(N_fwht)
    y_fwht = fwht(x_fwht, normalize=False)
    x_back = fwht(y_fwht, normalize=False) / N_fwht
    check(
        "FWHT is self‑inverse (H² = N·I)", np.allclose(x_fwht, x_back, atol=DEFAULT_TOL)
    )

    f = rng.standard_normal(N_fwht)
    g_r = rng.standard_normal(N_fwht)
    conv = xor_convolve(f, g_r)
    f_w = fwht(f, order="natural", normalize=True)
    g_w = fwht(g_r, order="natural", normalize=True)
    conv_w = fwht(conv, order="natural", normalize=True)
    check(
        "xor_convolve satisfies convolution theorem",
        np.allclose(np.sqrt(N_fwht) * f_w * g_w, conv_w, atol=DEFAULT_TOL),
    )
    seq_perm = sequency_permutation(n_fwht)
    check("sequency_permutation length matches 2^n", len(seq_perm) == N_fwht)
    gray = _gray_code_indices(n_fwht)
    check("_gray_code_indices length matches 2^n", len(gray) == N_fwht)
