"""
Thermodynamic classification of seed matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import eigvals

from .config import DEFAULT_RHO_MIN, DEFAULT_TOL


@dataclass
class SeedThermodynamics:
    """
    Information-geometric characterization of a seed matrix.

    The spectrum determines the "temperature" and entropy of the transform:
    - Conservative (unitary): max entropy, reversible, |λ|=1
    - Contractive: information loss, |λ|<1
    - Expansive: information generation, |λ|>1
    - Nilpotent: self-similar extinction, λ=0 eventually
    """

    spectral_radius: float
    min_eigenvalue_magnitude: float
    max_eigenvalue_magnitude: float
    is_unitary: bool
    is_conservative: bool  # |λ| = 1 for all λ
    is_contractive: bool  # |λ| < 1 for all λ
    is_expansive: bool  # |λ| > 1 for some λ
    is_nilpotent: bool  # S^k = 0 for some k
    lyapunov_exponent: float  # log max |λ|

    @classmethod
    def analyze(cls, S: np.ndarray, tol: float = DEFAULT_TOL) -> SeedThermodynamics:
        S = np.asarray(S, dtype=complex)
        eigs = eigvals(S)
        magnitudes = np.abs(eigs)
        rho = np.max(magnitudes)
        rho_min = np.min(magnitudes)
        rho_clamped = max(rho, DEFAULT_RHO_MIN)

        # Check unitarity (both left and right for completeness)
        is_unitary = np.allclose(
            S @ S.conj().T, np.eye(S.shape[0]), atol=tol
        ) and np.allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=tol)

        # Check nilpotency via characteristic polynomial.
        # np.poly returns [1, -trace, det, ...] so char_poly[0] is always 1.
        # Nilpotent iff all eigenvalues are 0, i.e. all non-leading coefficients vanish.
        char_poly = np.poly(S)
        is_nilpotent = np.all(np.abs(char_poly[1:]) < tol)

        return cls(
            spectral_radius=float(rho),
            min_eigenvalue_magnitude=float(rho_min),
            max_eigenvalue_magnitude=float(rho),
            is_unitary=is_unitary,
            is_conservative=bool(np.allclose(magnitudes, 1.0, atol=tol)),
            is_contractive=bool(np.all(magnitudes < 1 - tol)),
            is_expansive=bool(np.any(magnitudes > 1 + tol)),
            is_nilpotent=is_nilpotent,
            lyapunov_exponent=float(np.log(rho_clamped)),
        )

    def __str__(self) -> str:
        classification = []
        if self.is_nilpotent:
            classification.append("NILPOTENT (self-similar extinction)")
        elif self.is_unitary:
            classification.append("UNITARY/CONSERVATIVE (max entropy)")
        elif self.is_conservative:
            classification.append("CONSERVATIVE (|λ|=1, non-unitary)")
        elif self.is_contractive:
            classification.append("CONTRACTIVE (information loss)")
        elif self.is_expansive:
            classification.append("EXPANSIVE (information generation)")

        return f"Thermodynamics: {', '.join(classification)} | λ_max={self.spectral_radius:.4f} | h={self.lyapunov_exponent:.4f}"

    def __repr__(self) -> str:
        return f"SeedThermodynamics(spectral_radius={self.spectral_radius:.4f}, lyapunov_exponent={self.lyapunov_exponent:.4f}, is_unitary={self.is_unitary}, is_nilpotent={self.is_nilpotent})"


def nilpotency_index(S: np.ndarray, tol: float = DEFAULT_TOL) -> int | None:
    """Return smallest k ≥ 1 such that S^k = 0 (within tolerance), or None if not nilpotent."""
    d = S.shape[0]
    # Quick check using characteristic polynomial (nilpotent iff all eigenvalues zero)
    if not np.all(np.abs(np.poly(S)[1:]) < tol):
        return None
    # Compute powers up to d (Cayley-Hamilton guarantees index ≤ d)
    power = np.eye(d, dtype=S.dtype)
    for k in range(1, d + 1):
        power = power @ S
        if np.all(np.abs(power) < tol):
            return k
    return None  # should not happen if characteristic polynomial says nilpotent


# ------------------------------------------------------------------------------
# Standard seed constants
# ------------------------------------------------------------------------------

H2 = (1.0 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]])
"""Unitary Hadamard seed (2×2)."""

SEED_HADAMARD = H2
"""Unitary normalized Hadamard seed."""

SEED_ZETA = np.array([[1.0, 0.0], [1.0, 1.0]])
"""Zeta transform (prefix sum) seed."""

SEED_MOBIUS = np.array([[1.0, 0.0], [-1.0, 1.0]])
"""Möbius transform (inverse prefix sum) seed."""

SEED_LIFTING_P = np.array([[1.0, 0.5], [0.0, 1.0]])
"""Lifting scheme prediction step."""

SEED_LIFTING_U = np.array([[1.0, 0.0], [-0.25, 1.0]])
"""Lifting scheme update step."""

SEED_POLAR_GF2 = np.array([[1, 0], [1, 1]], dtype=np.uint8)
"""Polar code encoding seed over GF(2)."""

SEED_RULE90_GF2 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
"""Rule 90 cellular automaton seed over GF(2)."""

SEED_TROPICAL_VITERBI = np.array([[0.0, 1.0], [1.0, 0.0]])
"""Viterbi trellis step seed for tropical (min,+) semiring."""


def seed_rotation(theta: float) -> np.ndarray:
    """
    Rotation by angle theta (2×2 real orthogonal matrix).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        2×2 rotation matrix [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]].
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def seed_type_signature(S: np.ndarray) -> dict:
    """
    Full type signature of a 2×2 seed; forecasts behavior of S^{⊗n}.

    Returns a dictionary with keys:
      - "sum": sum of all entries
      - "trace": trace
      - "det": determinant
      - "frobenius2": squared Frobenius norm
      - "eigenvalues": array of eigenvalues
      - "is_unitary": bool
      - "spectral_radius": max |λ|
      - "notes": human‑readable summary
    """
    S = np.asarray(S, dtype=complex)
    s = S.sum()
    tr = np.trace(S)
    det = np.linalg.det(S)
    frob2 = float(np.real((S * S.conj()).sum()))
    eigs = np.linalg.eigvals(S)
    rho = float(np.max(np.abs(eigs)))
    is_unitary = bool(np.allclose(S @ S.conj().T, np.eye(2)))

    notes = []
    if is_unitary:
        notes.append("unitary → well-conditioned transform")
    if abs(tr) < 1e-10:
        notes.append("zero trace → constants annihilated")
    if rho > 1.0 + 1e-10:
        notes.append(f"spectral radius {rho:.3f} > 1 → exponentially ill-conditioned")
    if abs(det) < 1e-10:
        notes.append("singular seed → non-invertible transform")
    if abs(s - 2) < 1e-10:
        notes.append("sum=2 (Hadamard family)")
    if abs(s) < 1e-10:
        notes.append("sum=0 (annihilates constant inputs)")
    return {
        "sum": complex(s),
        "trace": complex(tr),
        "det": complex(det),
        "frobenius2": frob2,
        "eigenvalues": eigs,
        "is_unitary": is_unitary,
        "spectral_radius": rho,
        "notes": "; ".join(notes) if notes else "no special structure detected",
    }


def print_type_signature(S: np.ndarray, name: str = "S") -> None:
    """
    Pretty‑print the type signature of seed S.

    Parameters
    ----------
    S : np.ndarray
        2×2 seed matrix.
    name : str, optional
        Label for the seed in the output.
    """
    sig = seed_type_signature(S)
    w = 52
    print(f"\n{'─' * w}\n Seed type signature: {name}\n{'─' * w}")
    print(
        f" sum(S) = {sig['sum'].real:.6g}"
        + (f" + {sig['sum'].imag:.6g}j" if abs(sig["sum"].imag) > 1e-10 else "")
    )
    print(
        f" tr(S) = {sig['trace'].real:.6g}"
        + (f" + {sig['trace'].imag:.6g}j" if abs(sig["trace"].imag) > 1e-10 else "")
    )
    print(
        f" det(S) = {sig['det'].real:.6g}"
        + (f" + {sig['det'].imag:.6g}j" if abs(sig["det"].imag) > 1e-10 else "")
    )
    print(f" ||S||_F^2 = {sig['frobenius2']:.6g}")
    ev = sig["eigenvalues"]
    print(f" eigenvalues = [{ev[0]:.4g}, {ev[1]:.4g}]")
    print(f" spectral radius = {sig['spectral_radius']:.6g}")
    print(f" unitary = {sig['is_unitary']}")
    print(f" → {sig['notes']}\n{'─' * w}")


__all__ = [
    "SeedThermodynamics",
    "nilpotency_index",
    "H2",
    "SEED_HADAMARD",
    "SEED_ZETA",
    "SEED_MOBIUS",
    "SEED_LIFTING_P",
    "SEED_LIFTING_U",
    "SEED_POLAR_GF2",
    "SEED_RULE90_GF2",
    "SEED_TROPICAL_VITERBI",
    "seed_rotation",
    "seed_type_signature",
    "print_type_signature",
]
