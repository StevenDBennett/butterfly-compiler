"""
Utility functions for matrix operations and numerical linear algebra.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import matrix_power

try:
    from scipy.linalg import expm as _expm_scipy

    def expm(A: np.ndarray) -> np.ndarray:
        """
        Matrix exponential using SciPy if available.

        Falls back to Taylor series for small matrices if SciPy not installed.
        """
        return _expm_scipy(np.asarray(A, dtype=complex))
except ImportError:

    def expm(A: np.ndarray) -> np.ndarray:
        """
        Taylor series fallback for the matrix exponential.

        Accurate for small matrices (d <= 8) and moderate norm.
        """
        A = np.asarray(A, dtype=complex)
        d = A.shape[0]
        result = np.eye(d, dtype=complex)
        term = np.eye(d, dtype=complex)
        for k in range(1, 60):
            term = term @ A / k
            result += term
            if np.max(np.abs(term)) < 1e-16:
                break
        return result


def seed_power(S: np.ndarray, t: int) -> np.ndarray:
    """
    Compute S^t via repeated squaring (numpy.linalg.matrix_power).

    Parameters
    ----------
    S : np.ndarray
        Square matrix.
    t : int
        Non‑negative integer exponent.

    Returns
    -------
    np.ndarray
        S raised to power t.
    """
    return matrix_power(S, t)
