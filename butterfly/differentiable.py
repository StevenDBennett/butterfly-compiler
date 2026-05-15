"""
Differentiable butterfly transforms (forward/backward) for 2×2 seeds.

Implements the Adjoint Butterfly Theorem: (S^{⊗n})^T = (S^T)^{⊗n}, which yields
exact gradients w.r.t. every entry of every seed via reverse‑mode automatic
differentiation.
"""

from __future__ import annotations

import numpy as np


def butterfly_forward(
    x: np.ndarray,
    seeds: list[np.ndarray],
) -> tuple[np.ndarray, list]:
    """
    Forward pass of a multi‑seed butterfly with cached activations for reverse‑mode AD.

    Parameters
    ----------
    x : np.ndarray
        Input vector of length N = 2^n (real or complex).
    seeds : list[np.ndarray]
        List of 2×2 matrices; length must equal n = log₂(N).

    Returns
    -------
    y : np.ndarray
        Output vector (same length as x).
    cache : list
        List of tuples (u, v, h) storing activations for each level.

    Raises
    ------
    AssertionError
        If any seed is not 2×2, or length mismatches.
    """
    for s in seeds:
        if s.shape != (2, 2):
            raise ValueError(f"butterfly_forward requires 2×2 seeds, got {s.shape}")
    x = np.asarray(x).ravel().copy()
    # Promote to complex if input is complex, else use float64
    if np.iscomplexobj(x) or any(np.iscomplexobj(s) for s in seeds):
        x = x.astype(complex)
    else:
        x = x.astype(np.float64)
    N = x.size
    h = 1
    k = 0
    n = int(round(np.log2(N)))
    if (1 << n) != N:
        raise ValueError(f"N={N} must be a power of 2")
    if len(seeds) != n:
        raise ValueError(f"Need {n} seeds, got {len(seeds)}")

    cache: list = []
    while h < N:
        (a, b), (c, d) = seeds[k]
        x = x.reshape(-1, 2 * h)
        u = x[:, :h].copy()
        v = x[:, h:].copy()
        cache.append((u, v, h))
        x[:, :h] = a * u + b * v
        x[:, h:] = c * u + d * v
        x = x.ravel()
        h *= 2
        k += 1
    return x, cache


def butterfly_backward(
    grad_out: np.ndarray,
    seeds: list[np.ndarray],
    cache: list,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Reverse‑mode AD for multi‑seed butterfly.

    By the Adjoint Butterfly Theorem: (S^{⊗n})^T = (S^T)^{⊗n}
    so the backward pass is another butterfly with transposed seeds.

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient w.r.t. output (∂L/∂y).
    seeds : list[np.ndarray]
        Same seeds as used in forward pass (2×2 each).
    cache : list
        Cached activations from forward pass.

    Returns
    -------
    grad_in : np.ndarray
        Gradient w.r.t. input (∂L/∂x).
    seed_grads : list[np.ndarray]
        Gradient w.r.t. each seed matrix (∂L/∂S_k).
    """
    for s in seeds:
        if s.shape != (2, 2):
            raise ValueError(f"butterfly_backward requires 2×2 seeds, got {s.shape}")
    g = grad_out.copy().ravel()
    n = len(seeds)
    grad_dtype = complex if np.iscomplexobj(cache[0][0]) else np.float64
    seed_grads = [np.zeros((2, 2), dtype=grad_dtype) for _ in seeds]

    for k in range(n - 1, -1, -1):
        (a, b), (c, d) = seeds[k]
        u, v, h = cache[k]
        g = g.reshape(-1, 2 * h)
        gu = g[:, :h]
        gv = g[:, h:]

        # Gradient w.r.t. seed entries (rank‑1 outer products summed over blocks)
        seed_grads[k][0, 0] = (gu * u).sum()
        seed_grads[k][0, 1] = (gu * v).sum()
        seed_grads[k][1, 0] = (gv * u).sum()
        seed_grads[k][1, 1] = (gv * v).sum()

        # Propagate gradient through adjoint (conjugate-transpose) seed
        gu_in = a.conj() * gu + c.conj() * gv
        gv_in = b.conj() * gu + d.conj() * gv
        g[:, :h] = gu_in
        g[:, h:] = gv_in
        g = g.ravel()

    return g, seed_grads


__all__ = ["butterfly_forward", "butterfly_backward"]
