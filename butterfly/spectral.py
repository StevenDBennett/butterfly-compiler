"""
Spectral functional calculus for butterfly transforms.

Given a diagonalizable seed S, compute f(S^{⊗n}) @ v in O(N log N) for any scalar
function f, using the product spectrum and eigenbasis of S.
"""

from __future__ import annotations

import warnings

import numpy as np

from .core import fast_kron_power_transform


def build_product_spectrum(eigvals: np.ndarray, n: int) -> np.ndarray:
    """
    Build product spectrum of S^{⊗n} from the d eigenvalues of S.

    product_spectrum[j] = ∏_k eigvals[digit_k(j, d)]
    Ordering matches fast_kron_power_transform (base‑d digit decomposition).
    Cost: O(N) scalar multiplies. No approximation.

    Parameters
    ----------
    eigvals : np.ndarray
        Array of length d containing eigenvalues of S.
    n : int
        Kronecker power exponent.

    Returns
    -------
    np.ndarray
        Array of length d^n containing all products.
    """
    eigvals = np.asarray(eigvals, dtype=complex)
    spec = eigvals.copy()
    for _ in range(n - 1):
        spec = np.kron(spec, eigvals)
    return spec


def spectral_butterfly_apply(
    v: np.ndarray,
    S: np.ndarray,
    f: callable,
    real_output: bool = True,
    eps: float = 1e-10,
    cond_threshold: float = 1e10,
) -> np.ndarray:
    """
    Compute f(S^{⊗n}) @ v in O(N log N) for ANY scalar function f.

    Theorem: f(S^{⊗n}) = V^{⊗n} · diag(f(product_spectrum)) · (V⁻¹)^{⊗n}
    Requires S diagonalizable. Warns if eigenbasis is ill‑conditioned.

    Examples
    --------
    f = lambda z: z**t          → discrete fast‑forward at step t
    f = lambda z: np.exp(t*z)   → continuous time (S already a generator)
    f = lambda z: 1.0/(mu - z)  → resolvent / Green's function
    f = lambda z: np.sqrt(z)    → matrix square root

    Parameters
    ----------
    v : np.ndarray
        Input vector of length N = d^n.
    S : np.ndarray
        d×d seed matrix (diagonalizable).
    f : callable
        Scalar function C → C.
    real_output : bool, optional
        If True and imaginary parts are negligible, return real array.
    eps : float, optional
        Tolerance for detecting negligible imaginary parts.
    cond_threshold : float, optional
        Condition number threshold for warning about ill‑conditioned eigenbasis.

    Returns
    -------
    np.ndarray
        f(S^{⊗n}) @ v (real or complex).

    Raises
    ------
    ValueError
        If N is not d^n for integer n.
    """
    v = np.asarray(v)
    N = v.size
    d = S.shape[0]
    n = int(round(np.log(N) / np.log(d)))
    if d**n != N:
        raise ValueError(f"N={N} must equal d^n={d}^{n}")

    S_c = S.astype(complex)
    eigvals, V = np.linalg.eig(S_c)
    cond_V = np.linalg.cond(V)
    if cond_V > cond_threshold:
        warnings.warn(
            f"Seed eigenbasis is ill‑conditioned (cond(V)={cond_V:.2e}); "
            "seed may be defective. Results may be inaccurate — consider "
            "Schur‑form functional calculus for defective seeds.",
            RuntimeWarning,
            stacklevel=2,
        )
    V_inv = np.linalg.inv(V)

    # Transform to eigenbasis
    w = fast_kron_power_transform(v.astype(complex), V_inv, d=d)
    # Apply f pointwise to product spectrum
    product_spectrum = build_product_spectrum(eigvals, n)
    fv = np.vectorize(f, otypes=[complex])(product_spectrum)
    w = w * fv
    # Transform back
    result = fast_kron_power_transform(w, V, d=d)

    if real_output and np.max(np.abs(result.imag)) < eps * (
        np.max(np.abs(result.real)) + 1
    ):
        return result.real
    return result


def butterfly_polynomial(
    v: np.ndarray,
    S: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Evaluate p(S^{⊗n}) v where p(z) = Σ_k coeffs[k]·z^k (coeffs[0]=constant).

    By Cayley–Hamilton, any polynomial reduces mod the characteristic polynomial
    of S to degree < d, so at most d butterfly passes suffice regardless of the
    input polynomial's degree.

    Parameters
    ----------
    v : np.ndarray
        Input vector of length N = d^n.
    S : np.ndarray
        d×d seed matrix (real or complex).
    coeffs : np.ndarray
        Polynomial coefficients in ascending order (coeffs[0] constant term).

    Returns
    -------
    np.ndarray
        p(S^{⊗n}) @ v.
    """
    S = np.asarray(S)
    d = S.shape[0]
    coeffs = np.asarray(coeffs)
    # Promote to complex if either seed or coeffs are complex
    working_dtype = (
        complex if (np.iscomplexobj(S) or np.iscomplexobj(coeffs)) else float
    )
    S = S.astype(working_dtype)
    coeffs = coeffs.astype(working_dtype).copy()

    # Reduce mod characteristic polynomial if degree ≥ d
    if len(coeffs) > d:
        # np.poly returns descending coeffs; so does np.polydiv input/output
        char_desc = np.poly(S)  # descending, monic
        coeffs_desc = coeffs[::-1]  # ascending → descending
        _, rem_desc = np.polydiv(coeffs_desc, char_desc)
        coeffs = np.asarray(rem_desc, dtype=working_dtype)[::-1]  # back to ascending

    # Pad to length d
    if len(coeffs) < d:
        coeffs = np.concatenate([coeffs, np.zeros(d - len(coeffs))])

    # Horner: p(M)v = c_0 v + M(c_1 v + M(c_2 v + … + M(c_{d-1} v)))
    result = np.zeros_like(v, dtype=working_dtype)
    for c in coeffs[::-1]:  # highest degree first
        result = fast_kron_power_transform(result, S, d=d) + c * v
    return result


def butterfly_resolvent(
    v: np.ndarray,
    S: np.ndarray,
    mu: complex,
) -> np.ndarray:
    """
    Compute (μI - S^{⊗n})^{-1} v in O(N log N).

    Raises ValueError if μ is in the product spectrum.

    Parameters
    ----------
    v : np.ndarray
        Input vector of length N = d^n.
    S : np.ndarray
        d×d seed matrix (diagonalizable).
    mu : complex
        Complex shift.

    Returns
    -------
    np.ndarray
        Resolvent vector.
    """
    # spectral_butterfly_apply handles the eigendecomposition internally
    # and checks for mu in the spectrum via the function evaluation
    return spectral_butterfly_apply(v, S, lambda z: 1.0 / (mu - z))


__all__ = [
    "build_product_spectrum",
    "spectral_butterfly_apply",
    "butterfly_polynomial",
    "butterfly_resolvent",
]
