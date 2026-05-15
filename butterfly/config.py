"""
Configuration constants and dataclasses for the Butterfly Compiler.
"""

from dataclasses import dataclass

# Module-level constants for tolerance and numerical thresholds
DEFAULT_TOL = 1e-10
DEFAULT_ATOL = 1e-15
DEFAULT_RHO_MIN = 1e-300


@dataclass(slots=True)
class ButterflyConfig:
    """Configuration parameters for butterfly algorithms."""

    tol: float = DEFAULT_TOL
    atol: float = DEFAULT_ATOL
    rho_min: float = DEFAULT_RHO_MIN
    max_depth: int = 20


DEFAULT_CONFIG = ButterflyConfig()
