"""
Tests for differentiable butterfly transforms (forward/backward AD).
"""

import numpy as np

from butterfly.differentiable import butterfly_backward, butterfly_forward


def run_tests_differentiable(check, rng):
    """Tests for differentiable butterfly transforms."""
    print("\n── Differentiable Butterfly ────────────────────────────────────────")

    n = 3
    N = 1 << n
    seeds = [np.eye(2) + 0.1 * rng.standard_normal((2, 2)) for _ in range(n)]

    x = rng.standard_normal(N)
    y, cache = butterfly_forward(x, seeds)
    check("Forward pass produces output of correct length", len(y) == N)
    check("Forward pass produces cache of correct depth", len(cache) == n)

    grad_out = np.ones(N)
    grad_in, seed_grads = butterfly_backward(grad_out, seeds, cache)
    check("Backward pass produces input gradient of correct length", len(grad_in) == N)
    check("Backward pass produces seed gradients", len(seed_grads) == n)
    for g in seed_grads:
        check("Each seed gradient is 2x2", g.shape == (2, 2))

    y2, cache2 = butterfly_forward(x, seeds)
    check("Deterministic forward pass", np.allclose(y, y2))
    y_no_act, _ = butterfly_forward(x, [np.eye(2)] * n)
    check("Identity seeds pass through input", np.allclose(y_no_act, x))

    seeds_float = [s.astype(np.float64) for s in seeds]
    yf, cachef = butterfly_forward(x.astype(np.float64), seeds_float)
    check("Float64 forward pass works", np.allclose(yf, y, atol=1e-10))
