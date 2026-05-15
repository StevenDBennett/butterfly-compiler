"""
Operadic composition interface for butterfly transforms.

Provides `ButterflyOperad`, `MultiSeedOperad` and `CompiledButterfly` classes
that formalize the recursive structure of butterfly algorithms and enable
composition of different seeds at different levels (e.g., QFT construction).
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, ButterflyConfig
from .core import fast_kron_power_transform
from .semiring import GF2_SEMIRING, REAL_SEMIRING, Semiring
from .thermodynamics import nilpotency_index


class ButterflyOperad:
    """
    The butterfly as an operad: compositions of multi-input operations.

    Objects: Vector spaces V^{⊗n}
    Operations: S: V^{⊗d} → V^{⊗d} (the seed)
    Composition: grafting trees

    This formalizes the recursive structure and enables:
    - Automatic differentiation via operad duality
    - Verification of coherence laws
    - Composition of different seeds at different levels (multi-seed)
    """

    def __init__(
        self, seed: np.ndarray, arity: int = 2, semiring: Semiring = REAL_SEMIRING
    ):
        self.seed = np.asarray(seed)
        self.arity = arity
        self.semiring = semiring

    def compose(self, *sub_operads: ButterflyOperad) -> MultiSeedOperad:
        """
        Operadic composition: graft sub-operads into leaves.

        Validates that all sub-operads use the same semiring.
        Returns a MultiSeedOperad (which is not itself composable).

        If self has arity d and each sub has arity d, result has arity d^2.
        This is the vertical composition (depth increase).
        """
        for sub in sub_operads:
            if sub.semiring is not self.semiring:
                raise ValueError(
                    f"Sub-operad semiring {sub.semiring.name} does not match "
                    f"parent semiring {self.semiring.name}"
                )
        seeds = [self.seed] + [s.seed for s in sub_operads]
        return MultiSeedOperad(seeds, semiring=self.semiring)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return fast_kron_power_transform(
            x, self.seed, semiring=self.semiring, d=self.arity
        )

    def __repr__(self) -> str:
        return f"ButterflyOperad(seed_shape={self.seed.shape}, arity={self.arity}, semiring={self.semiring.name})"


class MultiSeedOperad:
    """
    Butterfly with a different 2×2 seed (and optional diagonal twiddle factors)
    at each level. Expressive enough to encode the QFT.

    At level k (k = 0, …, depth-1):
      1. If twiddles[k] is provided, multiply x element-wise by twiddles[k].
      2. Apply seed_k to each pair of blocks at stride h = 2^k.

    twiddles[k] has length N = 2^depth (acts on the full vector between levels).
    """

    def __init__(
        self,
        seeds: list[np.ndarray],
        twiddles: list[np.ndarray | None] | None = None,
        semiring: Semiring = REAL_SEMIRING,
    ):
        self.seeds = [np.asarray(s) for s in seeds]
        self.depth = len(seeds)
        self.semiring = semiring
        # GF(2) semiring requires integer dtypes
        if semiring is GF2_SEMIRING:
            for seed in self.seeds:
                if not np.issubdtype(seed.dtype, np.integer):
                    raise TypeError(
                        f"GF2 semiring requires integer dtype for seed, got {seed.dtype}"
                    )
        if twiddles is None:
            twiddles = [None] * self.depth
        if len(twiddles) != self.depth:
            raise ValueError(
                f"twiddles length ({len(twiddles)}) must equal depth ({self.depth})"
            )
        self.twiddles = [None if t is None else np.asarray(t) for t in twiddles]

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).copy()
        h = 1
        semiring = self.semiring

        # Buffer reuse optimization for real semiring
        if semiring is REAL_SEMIRING:
            # Check if we need complex dtype (any seed or twiddle complex)
            need_complex = False
            for seed, tw in zip(self.seeds, self.twiddles, strict=True):
                if np.iscomplexobj(seed) or (tw is not None and np.iscomplexobj(tw)):
                    need_complex = True
                    break
            # Allocate two buffers
            buf0 = x
            if need_complex and not np.iscomplexobj(buf0):
                buf0 = buf0.astype(complex)
            buf1 = np.empty_like(buf0)
            for k, seed in enumerate(self.seeds):
                d = seed.shape[0]
                tw = self.twiddles[k]
                if tw is not None:
                    buf0 *= tw  # element-wise multiplication
                # Reshape to 2D view
                src = buf0.reshape(-1, d * h)
                dst = buf1.reshape(-1, d * h)
                if d == 2:
                    # Fast path for 2x2
                    a, b = seed[0, 0], seed[0, 1]
                    c, d_ = seed[1, 0], seed[1, 1]
                    dst[:, 0:h] = a * src[:, 0:h] + b * src[:, h : 2 * h]
                    dst[:, h : 2 * h] = c * src[:, 0:h] + d_ * src[:, h : 2 * h]
                else:
                    # Generic d (unlikely for QFT)
                    for i in range(d):
                        dst[:, i * h : (i + 1) * h] = 0.0
                        for j in range(d):
                            dst[:, i * h : (i + 1) * h] += (
                                seed[i, j] * src[:, j * h : (j + 1) * h]
                            )
                # Swap buffers
                buf0, buf1 = buf1, buf0
                h *= d
            return buf0
        else:
            # Original generic implementation for other semirings
            for k, seed in enumerate(self.seeds):
                d = seed.shape[0]
                tw = self.twiddles[k]
                if tw is not None:
                    x = semiring.mul(x, tw)
                x = x.reshape(-1, d * h)
                blocks = [x[:, i * h : (i + 1) * h] for i in range(d)]
                new_x = np.full_like(x, semiring.zero)
                if semiring is REAL_SEMIRING and d == 2:
                    # Vectorized fast path for 2x2 real seed
                    a, b = seed[0, 0], seed[0, 1]
                    c, d_ = seed[1, 0], seed[1, 1]
                    new_x[:, 0:h] = a * blocks[0] + b * blocks[1]
                    new_x[:, h : 2 * h] = c * blocks[0] + d_ * blocks[1]
                else:
                    for i in range(d):
                        for j in range(d):
                            contrib = semiring.mul(seed[i, j], blocks[j])
                            new_x[:, i * h : (i + 1) * h] = semiring.add(
                                new_x[:, i * h : (i + 1) * h], contrib
                            )
                x = new_x.ravel()
                h *= d
            return x

    def __repr__(self) -> str:
        twiddle_info = sum(1 for t in self.twiddles if t is not None)
        return f"MultiSeedOperad(depth={self.depth}, seeds_shapes={[s.shape for s in self.seeds]}, twiddles={twiddle_info}/{self.depth}, semiring={self.semiring.name})"


class CompiledButterfly:
    """Pre‑compiled butterfly transform for repeated application."""

    def __init__(
        self,
        seed: np.ndarray,
        semiring: Semiring = REAL_SEMIRING,
        d: int | None = None,
        config: ButterflyConfig = DEFAULT_CONFIG,
    ):
        self.seed = np.asarray(seed)
        self.semiring = semiring
        self.d = d if d is not None else seed.shape[0]
        self.config = config

        # Validation
        if self.seed.ndim != 2 or self.seed.shape[0] != self.seed.shape[1]:
            raise ValueError(
                f"seed must be a square 2-D matrix; got shape {self.seed.shape}"
            )
        if self.d != self.seed.shape[0]:
            raise ValueError(
                f"d={self.d} doesn't match seed shape {self.seed.shape[0]}×{self.seed.shape[1]}"
            )
        # GF(2) semiring dtype validation
        if semiring is GF2_SEMIRING and not np.issubdtype(self.seed.dtype, np.integer):
            raise TypeError(
                f"GF2 semiring requires integer dtype for seed, got {self.seed.dtype}"
            )
        # Precompute nilpotency index
        self.nilpotent_index = nilpotency_index(self.seed, tol=config.tol)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the compiled butterfly transform to vector x."""
        return fast_kron_power_transform(x, self.seed, self.semiring, self.d)

    def __repr__(self) -> str:
        return f"CompiledButterfly(seed_shape={self.seed.shape}, d={self.d}, semiring={self.semiring.name}, nilpotent_index={self.nilpotent_index})"


__all__ = ["ButterflyOperad", "MultiSeedOperad", "CompiledButterfly"]
