"""
Continuous‑time fast‑forwarding and Dyson‑series perturbation for butterfly
dynamics on product grids.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .core import fast_kron_power_transform
from .utils import expm


def continuous_fast_forward(
    v0: np.ndarray,
    A: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Exact solution of dv/dt = A^{⊕n} v at time t.

    By Theorem 12.2: v(t) = (e^{tA})^{⊗n} v₀.
    Cost: O(d³) + O(N log N), independent of t.

    Parameters
    ----------
    v0 : np.ndarray
        Initial vector of length N = d^n.
    A : np.ndarray
        d×d generator matrix (real or complex).
    t : float
        Time.

    Returns
    -------
    np.ndarray
        v(t).
    """
    seed_t = expm(t * np.asarray(A, dtype=complex))
    result = fast_kron_power_transform(v0.astype(complex), seed_t)
    if np.isrealobj(v0) and np.max(np.abs(result.imag)) < 1e-10:
        return result.real
    return result


def separable_heat_solve(
    u0: np.ndarray,
    L_1d: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Exact solution of ∂_t u = (L^{⊕n}) u on a product grid.

    u(t) = (e^{tL})^{⊗n} u₀ — one butterfly pass, any t.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector of length N = d^n.
    L_1d : np.ndarray
        d×d generator matrix for one spatial dimension.
    t : float
        Time.

    Returns
    -------
    np.ndarray
        u(t).
    """
    return continuous_fast_forward(u0, L_1d, t)


def dyson_first_order(
    v0: np.ndarray,
    A: np.ndarray,
    B,
    t: float,
    n_quadrature: int = 8,
    warn_on_dense: bool = True,
) -> np.ndarray:
    """
    First‑order Dyson series for dv/dt = (A^{⊕n} + B) v.

    v(t) ≈ v_0(t) + v_1(t)
        v_0(t) = (e^{tA})^{⊗n} v₀
        v_1(t) = ∫₀ᵗ (e^{(t-s)A})^{⊗n} B (e^{sA})^{⊗n} v₀ ds

    B may be an N×N ndarray or a callable v → Bv (e.g., rank‑r form).
    Error: O(‖B‖² t²). Use dyson_rank1_correction for rank‑1 B.
    Cost: O(n_quadrature · N log N) plus O(N · rank(B)) per node.

    Parameters
    ----------
    v0 : np.ndarray
        Initial vector of length N = d^n.
    A : np.ndarray
        d×d generator matrix.
    B : np.ndarray or callable
        Perturbation matrix (N×N) or linear operator.
    t : float
        Time.
    n_quadrature : int, optional
        Number of Gauss–Legendre nodes for quadrature.

    Returns
    -------
    np.ndarray
        Approximate v(t) (real if inputs real).
    """
    v_zero = continuous_fast_forward(v0, A, t)
    nodes, weights = leggauss(n_quadrature)
    s_vals = 0.5 * t * (nodes + 1)
    w_vals = 0.5 * t * weights
    if not callable(B) and warn_on_dense and B.ndim == 2:
        import warnings

        warnings.warn(
            "Dense B matrix in dyson_first_order: matmul is O(N\u00b2) per "
            "quadrature node, dominating the O(N log N) butterfly cost. "
            "Use a callable (rank-r form) for O(N log N) scaling.",
            RuntimeWarning,
            stacklevel=2,
        )
    v_first = np.zeros_like(v0, dtype=complex)
    for s, w in zip(s_vals, w_vals, strict=True):
        inner = continuous_fast_forward(v0, A, s)
        perturbed = B(inner) if callable(B) else np.asarray(B) @ inner
        outer = continuous_fast_forward(perturbed, A, t - s)
        v_first = v_first + w * outer
    result = v_zero + v_first
    if np.isrealobj(v0) and np.max(np.abs(result.imag)) < 1e-10:
        return result.real
    return result


def dyson_rank1_correction(
    v0: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    w_vec: np.ndarray,
    t: float,
    n_quadrature: int = 8,
) -> np.ndarray:
    """
    Rank‑1 Dyson correction with B = u ⊗ w_vec^T. Cost O(n_quad · N log N).

    Parameters
    ----------
    v0 : np.ndarray
        Initial vector of length N.
    A : np.ndarray
        d×d generator matrix.
    u, w_vec : np.ndarray
        Vectors of length N defining rank‑1 perturbation B = u w_vec^T.
    t : float
        Time.
    n_quadrature : int, optional
        Number of Gauss–Legendre nodes.

    Returns
    -------
    np.ndarray
        Approximate v(t).
    """

    def _b_rank1(v):
        return np.dot(w_vec, v) * u

    return dyson_first_order(v0, A, _b_rank1, t, n_quadrature)


__all__ = [
    "continuous_fast_forward",
    "separable_heat_solve",
    "dyson_first_order",
    "dyson_rank1_correction",
]
