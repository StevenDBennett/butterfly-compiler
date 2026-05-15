"""
Butterfly Compiler package.

Provides fast Kronecker-power transforms, thermodynamic classification,
semiring abstraction, solvability analysis, block-butterfly compilation,
symbolic fast-forwarding, operadic composition, QFT/NTT implementations,
and quantum circuit export.
"""

__version__ = "1.0"

from .block import BlockSeed, block_butterfly_transform
from .config import (
    DEFAULT_ATOL,
    DEFAULT_CONFIG,
    DEFAULT_RHO_MIN,
    DEFAULT_TOL,
    ButterflyConfig,
)
from .core import (
    fast_forward,
    fast_gf2_transform,
    fast_kron_power_transform,
    fast_multi_seed,
    fast_tropical_transform,
    naive_intt,
    naive_ntt,
    rule90_fast_forward,
)
from .differentiable import butterfly_backward, butterfly_forward
from .discovery import (
    enumerate_seeds_with_properties,
    enumerate_solvable_seeds,
    print_discovered_seeds,
)
from .dynamics import (
    continuous_fast_forward,
    dyson_first_order,
    dyson_rank1_correction,
    separable_heat_solve,
)
from .export_qasm import export_qft_qasm, export_unitary_seed_qasm
from .fwht import fwht, sequency_permutation, xor_convolve
from .ntt import ntt_butterfly
from .operad import ButterflyOperad, CompiledButterfly, MultiSeedOperad
from .qft import (
    fft_seeds,
    qft_butterfly,
    qft_direct,
    qft_direct_combined,
)
from .semiring import (
    GF2_SEMIRING,
    LOG_SEMIRING,
    MOD_NTT_998244353,
    REAL_SEMIRING,
    TROPICAL_MAX,
    TROPICAL_MIN,
    Semiring,
    modular_semiring,
)
from .solvability import lie_closure, print_solvability, solvability_series
from .spectral import (
    build_product_spectrum,
    butterfly_polynomial,
    butterfly_resolvent,
    spectral_butterfly_apply,
)
from .symbolic import (
    SymbolicMatrix,
    symbolic_kronecker_power,
    symbolic_matrix_power,
)
from .thermodynamics import (
    H2,
    SEED_HADAMARD,
    SEED_LIFTING_P,
    SEED_LIFTING_U,
    SEED_MOBIUS,
    SEED_POLAR_GF2,
    SEED_RULE90_GF2,
    SEED_TROPICAL_VITERBI,
    SEED_ZETA,
    SeedThermodynamics,
    nilpotency_index,
    print_type_signature,
    seed_rotation,
    seed_type_signature,
)
from .utils import expm, seed_power

# Export key symbols from submodules (will be populated as modules are loaded)
__all__ = [
    # Package version
    "__version__",
    # Configuration
    "DEFAULT_TOL",
    "DEFAULT_ATOL",
    "DEFAULT_RHO_MIN",
    "ButterflyConfig",
    "DEFAULT_CONFIG",
    # Thermodynamic classification
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
    # Utilities
    "expm",
    "seed_power",
    # Discovery
    "enumerate_solvable_seeds",
    "enumerate_seeds_with_properties",
    "print_discovered_seeds",
    # Semirings
    "Semiring",
    "REAL_SEMIRING",
    "TROPICAL_MIN",
    "TROPICAL_MAX",
    "GF2_SEMIRING",
    "LOG_SEMIRING",
    "modular_semiring",
    "MOD_NTT_998244353",
    # Core transforms
    "fast_kron_power_transform",
    "naive_ntt",
    "naive_intt",
    "fast_forward",
    "rule90_fast_forward",
    "fast_multi_seed",
    "fast_gf2_transform",
    "fast_tropical_transform",
    # Solvability analysis
    "solvability_series",
    "lie_closure",
    "print_solvability",
    # Block butterfly
    "BlockSeed",
    "block_butterfly_transform",
    # Symbolic fast-forwarding
    "SymbolicMatrix",
    "symbolic_matrix_power",
    "symbolic_kronecker_power",
    # Spectral calculus
    "build_product_spectrum",
    "spectral_butterfly_apply",
    "butterfly_polynomial",
    "butterfly_resolvent",
    # Dynamics
    "continuous_fast_forward",
    "separable_heat_solve",
    "dyson_first_order",
    "dyson_rank1_correction",
    # Operadic composition
    "ButterflyOperad",
    "MultiSeedOperad",
    "CompiledButterfly",
    # Differentiable
    "butterfly_forward",
    "butterfly_backward",
    # QFT
    "qft_butterfly",
    "fft_seeds",
    "qft_direct",
    "qft_direct_combined",
    # FWHT
    "fwht",
    "xor_convolve",
    "sequency_permutation",
    # NTT
    "ntt_butterfly",
    # Quantum circuit export
    "export_qft_qasm",
    "export_unitary_seed_qasm",
]
