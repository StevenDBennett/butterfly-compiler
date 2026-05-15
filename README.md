# Butterfly Compiler

**Version 1.0** · **Research project** · **Python 3.13.7 + NumPy 2.4.4**

A comprehensive mathematical library implementing fast Kronecker‑power transforms (butterfly algorithms) over arbitrary semirings, with thermodynamic classification of seed matrices, Lie‑algebra solvability analysis, block‑butterfly hierarchical compilation, symbolic fast‑forwarding, operadic composition, QFT/NTT implementations, and quantum circuit export.

## Overview

The Butterfly Compiler is a research codebase exploring the mathematical foundations of butterfly transforms—fast algorithms for applying Kronecker powers of small matrices (`S^(⊗n)`) to vectors in O(N log N) time. The library provides:

- **Semiring abstraction**: Transform over real numbers, tropical (min/max), GF(2), log‑domain, and modular arithmetic (NTT)
- **Thermodynamic classification**: Characterizes 2×2 seeds via eigenvalue spectrum (unitary/conservative/contractive/expansive/nilpotent)
- **Solvability engine**: Lie‑algebra derived series (Beth’s theorem) to guarantee existence of O(N log N) butterfly algorithms
- **Block‑butterfly**: Hierarchical compilation of larger structured matrices as nested butterfly circuits
- **Symbolic layer**: Polynomial‑based fast‑forwarding of high Kronecker powers without explicit expansion
- **Operadic interface**: Compositional algebra for combining butterfly operations
- **QFT/NTT**: Quantum Fourier transform via Van Loan factorization (matches NumPy FFT to 10⁻¹⁵), number‑theoretic transform
- **Quantum circuit export**: QASM generation for QFT and unitary seeds (including qudit circuits)
- **Performance benchmarks**: Built-in performance benchmarks

All algorithms are unified under thermodynamic/solvability constraints derived from the literature (Fino‑Algazi, Diaconis‑Rockmore, etc.).

## Installation

```bash
git clone https://github.com/butterfly-compiler/butterfly-compiler.git
cd butterfly-compiler
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install numpy
```

The library has only one dependency: NumPy ≥ 2.0.

## Quick Start

```python
import numpy as np
from butterfly import (
    fast_kron_power_transform, REAL_SEMIRING, TROPICAL_MIN,
    SeedThermodynamics, solvability_series, qft_butterfly
)

# Apply Hadamard transform to a random vector
x = np.random.randn(64)
y = fast_kron_power_transform(x, np.array([[1, 1], [1, -1]]) / np.sqrt(2), REAL_SEMIRING)

# Classify a seed matrix
thermo = SeedThermodynamics.analyze(np.array([[0, 1], [1, 0]]))
print(f"Unitary: {thermo.is_unitary}, Conservative: {thermo.is_conservative}")

# Check solvability (Beth's theorem)
S = np.array([[1, 1], [0, 1]])  # upper triangular
result = solvability_series(S)
print(f"Solvable: {result['is_solvable']}, series: {result['series']}")

# Compute QFT (matches numpy.fft to 1e-15)
x = np.random.randn(16) + 1j * np.random.randn(16)
qft_apply = qft_butterfly(4, include_swaps=True)
y_qft = qft_apply(x)
y_np = np.fft.fft(x, norm="ortho")
print(f"Max error: {np.max(np.abs(y_qft - y_np)):.2e}")
```

Run the self‑test suite:
```bash
python butterfly_compiler.py --all
```

## API Reference

The package is organized into modular submodules; all public symbols are re‑exported from `butterfly`. The following tables list the main functions and classes by module.

### 1. `butterfly.core` – Core butterfly transforms

| Function | Description |
|----------|-------------|
| `fast_kron_power_transform(x, S, semiring=REAL_SEMIRING)` | O(N log N) application of `S^(⊗n)` to vector `x` (size `N = d^n`) |
| `fast_gf2_transform(x, S)` | Specialized GF(2) semiring transform |
| `fast_tropical_transform(x, S)` | Specialized tropical (min‑plus) transform |
| `fast_multi_seed(x, seeds)` | Apply multiple seeds in sequence (operadic composition) |
| `fast_forward(x, seed, t, semiring=REAL_SEMIRING, d=None)` | Fast‑forwarding via `(seed^t)^(⊗n)`, O(d³ log t) + O(N log N) |
| `rule90_fast_forward(x, t)` | Rule‑90 cellular automaton over GF(2); even t > log₂ N → zero |
| `naive_ntt(x, omega, p)` | Naïve O(N²) number‑theoretic transform (reference) |
| `naive_intt(X, omega, p, scale=True)` | Naïve inverse NTT |

### 2. `butterfly.semiring` – Semiring abstraction

| Class/Function | Description |
|----------------|-------------|
| `Semiring(add, mul, zero, one, matvec)` | Abstract semiring with addition `⊕`, multiplication `⊗`, identity elements |
| `REAL_SEMIRING` | Standard real arithmetic (`+`, `×`) |
| `TROPICAL_MIN` | Tropical (min‑plus) semiring (`min`, `+`) |
| `TROPICAL_MAX` | Tropical (max‑plus) semiring (`max`, `+`) |
| `GF2_SEMIRING` | GF(2) arithmetic (`xor`, `and`) |
| `LOG_SEMIRING` | Log‑domain (`logaddexp`, `+`) |
| `modular_semiring(p)` | Modular arithmetic over ℤ/p (`+ mod p`, `× mod p`) |
| `MOD_NTT_998244353` | Pre‑defined NTT‑friendly modulus 998244353 |

### 3. `butterfly.thermodynamics` – Seed classification

| Class/Function | Description |
|----------------|-------------|
| `SeedThermodynamics(S, tol=DEFAULT_TOL)` | Classifies 2×2 seeds via eigenvalue spectrum |
| `nilpotency_index(S)` | Returns smallest k such that `S^k = 0` (or `None` if not nilpotent) |
| `seed_type_signature(S)` | Returns string classification (unitary/conservative/contractive/expansive/nilpotent) |
| `print_type_signature(S, name="S")` | Pretty‑prints classification |
| `H2` | Normalized Hadamard matrix `[[1,1],[1,-1]]/√2` |
| `SEED_HADAMARD`, `SEED_ZETA`, `SEED_MOBIUS`, `SEED_LIFTING_P`, `SEED_LIFTING_U`, `SEED_POLAR_GF2`, `SEED_RULE90_GF2`, `SEED_TROPICAL_VITERBI`, `seed_rotation(theta)` | Pre‑defined standard seeds |

### 4. `butterfly.solvability` – Lie‑algebra analysis

| Function | Description |
|----------|-------------|
| `solvability_series(S, max_depth=20, tol=DEFAULT_TOL, include_adjoint=False, method='svd', random_seed=None)` | Computes derived series of Lie algebra generated by `S` (and optionally `S†`). Methods: `'svd'` (robust), `'incremental'` (faster), `'adjoint'` (O(d⁶)), `'chevalley'` (O(d⁵) randomized) |
| `lie_closure(generators, tol=1e-10, max_iter=100)` | Smallest Lie subalgebra containing the generators |
| `print_solvability(S, name="S")` | Pretty‑prints solvability verdict |

### 5. `butterfly.block` – Block‑butterfly hierarchical compilation

| Class/Function | Description |
|----------------|-------------|
| `BlockSeed(blocks, tol=DEFAULT_TOL)` | Represents a block‑structured seed matrix for hierarchical butterfly compilation |
| `block_butterfly_transform(x, block_seed, semiring=REAL_SEMIRING)` | Applies block‑butterfly transform to vector `x` |

### 6. `butterfly.symbolic` – Symbolic fast‑forwarding

| Class/Function | Description |
|----------------|-------------|
| `SymbolicMatrix(coeffs, max_degree=100, n_vars=None, shape=None)` | Matrix with polynomial entries (sparse dict `{(i,j,exp): coeff}`) |
| `SymbolicMatrix.from_matrix(M, n_vars=2, max_degree=100)` | Creates `SymbolicMatrix` from numeric numpy array |
| `symbolic_kronecker_power(S, n, row_tuple, col_tuple)` | Computes a single entry `(S^(⊗n))_{row, col}` without expanding |
| `symbolic_matrix_power(S, n, entry=(0, 0))` | Computes `(S^n)_{i,j}` — single entry of ordinary matrix power |

### 7. `butterfly.operad` – Operadic composition

| Class | Description |
|-------|-------------|
| `ButterflyOperad(seed, arity=2, semiring=REAL_SEMIRING)` | Single‑seed butterfly operation |
| `MultiSeedOperad(seeds, twiddles=None, semiring=REAL_SEMIRING)` | Sequence of seeds (with optional twiddle factors) |
| `CompiledButterfly(seed, semiring=REAL_SEMIRING, d=None, config=DEFAULT_CONFIG)` | Pre‑validated butterfly transform for repeated application; precomputes nilpotency index |
| `ButterflyOperad.compose(*sub_operads)` | Returns `MultiSeedOperad` (validates matching semirings) |

### 8. `butterfly.qft` – Quantum Fourier transform

| Function | Description |
|----------|-------------|
| `qft_butterfly(n_qubits, include_swaps=True, return_operad=False, inverse=False)` | Returns QFT apply function (callable); `inverse=True` gives inverse QFT (conjugated twiddles). If `return_operad=True` returns `(operad, apply_fn)`. Matches `numpy.fft` to 1e‑15 |
| `qft_direct(x, include_swaps=True)` | Direct optimised QFT loop (avoids `MultiSeedOperad` overhead) |
| `qft_direct_combined(x, include_swaps=True)` | QFT with fused twiddle and butterfly operations |
| `fft_seeds(n)` | Returns list of n complex 2×2 seeds for Cooley–Tukey DFT_{2^n} braid |
| `_bit_reverse_indices(n_bits)` | Returns bit‑reversal permutation indices for length `2^n` (internal) |
| `_bit_reverse_permutation(x, n_bits)` | Applies bit‑reversal permutation to array `x` (internal) |

### 9. `butterfly.ntt` – Number‑theoretic transform

| Function | Description |
|----------|-------------|
| `ntt_butterfly(n, p, omega, inverse=False, include_swaps=True, scale=True)` | Constructs `(operad, apply_fn)` for NTT over ℤ/p (size N=2ⁿ) |

### 10. `butterfly.fwht` – Fast Walsh‑Hadamard transform

| Function | Description |
|----------|-------------|
| `fwht(x, order='sequency', normalize=True)` | Fast Walsh‑Hadamard transform (natural, sequency, or Paley order) |
| `xor_convolve(f, g)` | XOR convolution via FWHT (`(f ∗ g)[k] = Σ_{i⊕j=k} f[i] g[j]`) |
| `sequency_permutation(n_bits)` | Returns permutation that maps natural to sequency order |
| `_gray_code_indices(n_bits)` | Returns Gray‑code indices for sequency ordering (internal) |

### 11. `butterfly.spectral` – Spectral calculus

| Function | Description |
|----------|-------------|
| `spectral_butterfly_apply(v, S, f, real_output=True, eps=1e-10, cond_threshold=1e10)` | Computes `f(S^(⊗n)) @ v` for ANY scalar function `f` (requires diagonalizable `S`) |
| `build_product_spectrum(eigvals, n)` | Builds product spectrum of `S^(⊗n)` from d eigenvalues of `S` |
| `butterfly_polynomial(v, S, coeffs)` | Evaluates `p(S^(⊗n)) v` where `p(z) = Σ coeffs[k]·z^k` |
| `butterfly_resolvent(v, S, mu)` | Computes resolvent `(μI − S^(⊗n))⁻¹ v` |

### 12. `butterfly.dynamics` – Dynamical systems

| Function | Description |
|----------|-------------|
| `continuous_fast_forward(v0, A, t)` | Exact solution of `dv/dt = A^(⊕n) v`: `v(t) = (e^{tA})^(⊗n) v0` |
| `separable_heat_solve(u0, L_1d, t)` | Wrapper for heat equation `∂_t u = L^(⊕n) u` on product grid |
| `dyson_first_order(v0, A, B, t, n_quadrature=8, warn_on_dense=True)` | First‑order Dyson series `v(t) ≈ v₀ + v₁` for perturbed generator `A^(⊕n) + B` |
| `dyson_rank1_correction(v0, A, u, w_vec, t, n_quadrature=8)` | Rank‑1 Dyson correction with `B = u w_vec^T` |

### 13. `butterfly.differentiable` – Automatic differentiation

| Function | Description |
|----------|-------------|
| `butterfly_forward(x, seeds)` | Forward pass of multi‑seed butterfly transform |
| `butterfly_backward(grad_out, seeds, cache)` | Backward pass (gradients w.r.t. seeds) |

### 14. `butterfly.export_qasm` – Quantum circuit export

| Function | Description |
|----------|-------------|
| `export_qft_qasm(n_qubits, include_swaps=True)` | Generates OpenQASM 2.0 code for n‑qubit QFT (returns string) |
| `export_unitary_seed_qasm(seed, n_qubits=1, fmt='qasm')` | Exports unitary seed as QASM circuit; for d>2 embeds into next power‑of‑two (qudit) |

### 15. `butterfly.discovery` – Automatic seed discovery

| Function | Description |
|----------|-------------|
| `enumerate_solvable_seeds(entry_values=(-1,0,1), d=2, tol=DEFAULT_TOL, max_samples=None, random_state=None)` | Brute‑force / random enumeration of butterfly‑compilable (solvable) seeds |
| `enumerate_seeds_with_properties(entry_values=(-1,0,1), d=2, *, unitary=None, conservative=None, contractive=None, expansive=None, nilpotent=None, tol=DEFAULT_TOL, max_samples=None, random_state=None)` | Filter seeds by keyword‑only thermodynamic property flags |
| `print_discovered_seeds(entry_values=(-1,0,1), d=2, tol=DEFAULT_TOL, max_samples=None, random_state=None)` | Pretty‑prints discovered seeds with classification |

### 16. `butterfly.utils` – Utilities

| Function | Description |
|----------|-------------|
| `expm(S)` | Matrix exponential via scaling‑and‑squaring |
| `seed_power(S, power)` | Computes `S^power` via repeated squaring |

### 17. `butterfly.config` – Configuration

| Constant/Class | Description |
|----------------|-------------|
| `DEFAULT_TOL = 1e-10` | Default numerical tolerance |
| `DEFAULT_ATOL = 1e-15` | Default absolute tolerance |
| `DEFAULT_RHO_MIN = 1e-300` | Minimum spectral radius for classification |
| `ButterflyConfig(tol=DEFAULT_TOL, atol=DEFAULT_ATOL, rho_min=DEFAULT_RHO_MIN, max_depth=20, include_adjoint=False)` | Configuration dataclass |
| `DEFAULT_CONFIG` | Default configuration instance |

## Command‑Line Interface

The wrapper script `butterfly_compiler.py` provides command‑line access to key functionality:

```bash
# Run full test suite + benchmarks
python butterfly_compiler.py --all

# Run benchmarks only
python butterfly_compiler.py --benchmark

# Print discovered solvable seeds (d=2, entries {-1,0,1})
python butterfly_compiler.py --discover
```

## Performance

The library is optimized for practical use:

- **Real semiring**: Vectorized `einsum` fast path (O(N log N) with low constant)
- **Tropical/log/GF(2)**: Vectorized fast paths within 2–3× of real semiring
- **QFT**: 3–6× slower than NumPy FFT (vs. 5–12× before optimization)
- **Solvability analysis**:  
  - `adjoint` method: O(d⁶), 10× speedup for d=6  
  - `chevalley` method: O(d⁵), 2–6× speedup for d=4–10
- **Memory**: No leaks confirmed via `tracemalloc` profiling

All transforms have numerical accuracy verified by hypothesis‑based property tests.

## Mathematical Background

### Beth's Theorem
A seed matrix `S` admits an O(N log N) butterfly algorithm **iff** the Lie algebra generated by `S` alone (not including `S†`) is solvable. The function `solvability_series` implements this check via derived‑series computation.

### Thermodynamic Classification
For 2×2 seeds, the eigenvalue spectrum determines algorithmic properties:
- **Unitary**: `|λ| = 1` (norm‑preserving, e.g., Hadamard)
- **Conservative**: `|λ₁|·|λ₂| = 1` (volume‑preserving)
- **Contractive**: `ρ(S) < 1` (spectral radius < 1)
- **Expansive**: `ρ(S) > 1`
- **Nilpotent**: `S^k = 0` for some k (eigenvalues all zero)

### Semiring Abstraction
Butterfly transforms generalize to any semiring `(⊕, ⊗, 0, 1)` where:
- `⊕` associative, commutative, identity `0`
- `⊗` associative, identity `1`, distributes over `⊕`
- `0 ⊗ a = a ⊗ 0 = 0`

Implemented semirings: real, tropical min/max, GF(2), log‑domain, modular (NTT).

## Experiments

The `experiments/` directory contains twelve experiments validating the Lohmiller–Slotine
claim that Schrödinger wave functions can be constructed exactly from classical action branches
[6]. Each experiment tests a specific aspect of the butterfly-compilable decomposition
$\psi = \sum_j \sqrt{\rho_j}\, e^{i\phi_j/\hbar}$:

### exp1 — Benchmark seed analyses (initial exploration)
### exp2 — Lohmiller–Slotine double-slit bridge
  Tests the core claim: the double-slit wavefunction decomposes into two independent classical
  branches, each propagated by the same butterfly QFT. Interference fringes emerge from the
  coherent sum of branches. Linearity verified to 0.9999.

### exp3 — Multi-slit diffraction
  Generalizes the branch decomposition to N slits (N = 2, 3, 4, 5, 7). Linearity holds for any
  number of branches, and the butterfly QFT compiles all branches simultaneously in one
  O(N log N) pass.

### exp4 — Quantum harmonic oscillator (split-step butterfly QFT)
  Propagates a trapped wavepacket via Strang-split butterfly QFT over a full oscillation period.
  Norm conserved to machine precision; classical trajectory $\langle x \rangle(t) = x_0 \cos(\omega t)$
  matches to $10^{-4}$. Both kinetic and potential generators are solvable (Beth's theorem),
  certifying the O(N log N) compilation.

### exp5 — Dyson series for coupled 2-qubit system
  Tests `continuous_fast_forward` and `dyson_first_order` from the dynamics module on a
  2-qubit system with Ising coupling. Validates perturbative corrections to product-system
  evolution; Dyson error scales as O(J²t²).

### exp6 — Bohmian quantum potential diagnostic
  Computes the quantum potential $Q = -\hbar^2/(2m) \cdot \nabla^2|\psi| / |\psi|$ from
  propagated wavefunctions. Maps quantum-dominated regions (interference fringes) vs classical
  regions (tails, free propagation). For the harmonic oscillator ground state, verifies
  $Q = -V(x)$ (Bohm's equilibrium).

### exp7 — Schrödinger cat state in anharmonic oscillator
  Adds a quartic anharmonic term $\lambda x^4$ to the harmonic potential. A displaced coherent
  state evolves into a cat state (two macroscopically separated branches) at half the revival time.
  Demonstrates butterfly compilation of extreme branch proliferation.

### exp8 — 2D heat equation on product grid
  Solves $\partial_t u = \alpha \nabla^2 u$ on a 2D periodic domain using
  `separable_heat_solve` from `butterfly.dynamics`. The heat kernel factorizes as a Kronecker
  product of 1D kernels: $u(t) = (e^{t\alpha L_{1d}})^{\otimes 2} \ u_0$, computed in one
  O(N log N) butterfly pass. Validates interior L2 error vs analytic Gaussian spreading,
  mass conservation, linearity, and t=0 matching.

### exp9 — Variational ground state via differentiable butterfly
  Optimizes 2×2 seed entries via gradient descent on the Rayleigh quotient
  $E = \langle\psi|H|\psi\rangle$ using `butterfly_forward`/`butterfly_backward` from
  `butterfly.differentiable`. Adam optimization converges to the harmonic oscillator ground
  state with overlap > 0.9999999. Validates AD gradients against finite differences,
  energy convergence, and gradient norm reduction.

### exp10 — Quantum particle in a box (method of images)
  Propagates a Gaussian wavepacket in an infinite square well via the method of images
  with QFT butterfly on a doubled domain $[-L, L]$. The odd extension enforces Dirichlet
  BCs exactly. Validates norm conservation, hard-wall reflection, round-trip return, and
  odd-symmetry preservation to machine precision.

### exp11 — Quantum tunnelling through a rectangular barrier
  Propagates a Gaussian wavepacket with energy $E < V_0$ toward a rectangular barrier
  using split-step butterfly QFT. The wavepacket splits into reflected (84%) and
  transmitted (16%) components. The measured transmission matches the momentum-weighted
  plane-wave average to 2%, and tunnelling emerges from complex-valued classical action
  branches (Example 3.8 of Lohmiller–Slotine).

### exp12 — Aharonov-Bohm effect via butterfly QFT
  Imposes a relative phase $\alpha = q\Phi/\hbar$ between the two branches of a
  double-slit interference pattern, simulating an enclosed magnetic flux. The
  interference fringes shift by exactly $\alpha$, with the FFT phase tracking
  $\alpha$ linearly (slope 1.0003). The shift is $2\pi$-periodic (gauge invariance),
  and the single-slit envelope is unchanged. Validates the Lohmiller–Slotine claim
  (Example 3.6) that the vector potential enters the classical action as $q\int A\cdot dx$.

Run any experiment:
```bash
source venv/bin/activate
PYTHONPATH=. python experiments/exp2_ls_bridge.py
PYTHONPATH=. python experiments/exp7_cat_state.py
```

## Development

### Code Style
- PEP 8 conventions, NumPy‑style docstrings
- Type hints throughout
- Ruff for formatting/linting (`ruff check .`, `ruff format .`)

### Testing
- 233/233 unit tests passing (core, edge cases, property‑based)
- Hypothesis‑based property verification

### Virtual Environment
```bash
source venv/bin/activate          # Activate
python butterfly_compiler.py --all      # Run tests
ruff check --fix .               # Lint
```

## References

1. **Beth, T.** (1984). *Algebraic algorithms for the discrete Fourier transform*.  
2. **Diaconis, P., & Rockmore, D.** (1990). *Efficient computation of the Fourier transform on finite groups*.  
3. **Fino, A., & Algazi, V. R.** (1976). *Unified matrix treatment of the fast Walsh–Hadamard transform*.  
4. **Van Loan, C.** (1992). *Computational frameworks for the fast Fourier transform*.  
5. **Chevalley, C., & Eilenberg, S.** (1948). *Cohomology theory of Lie groups and Lie algebras*.
6. **Lohmiller, W., & Slotine, J-J.** (2026). *On computing quantum waves exactly from classical action*. Proc. R. Soc. A 482: 20250413.

## License

Research code – see repository for licensing information.

---
*Butterfly Compiler v1.0 · 2026‑05‑15*