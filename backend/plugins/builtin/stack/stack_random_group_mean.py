"""Built-in node — stack.random_group_mean (Random K-group means) — Stack.

Randomly partition a stack into `k` roughly-equal groups and return the
per-group mean as a stack of `k` images. Intended as the initial-reference
seed for iterative multi-reference classification: averaging ~N/k frames
per group already smooths out noise far better than a single random frame,
and the groups statistically contain a mix of all true classes, so no
single bucket collapses onto one class right out of the gate.

Pair with ``align.ap_mr`` + ``stack.class_mean`` inside a DO loop.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param
from backend.nodes import h, d


def _fn_random_group_mean(stack, k: int = 4, samples_per_group: int = 1,
                          seed: int = 0):
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3:
        raise ValueError(
            f"stack.random_group_mean: expected a 3D stack, got shape {s.shape}"
        )
    n, ny, nx = s.shape
    k_int = max(1, int(k))
    m = max(1, int(samples_per_group))
    rng = np.random.default_rng(int(seed))
    # Pick k * m DISTINCT frames at random (if the stack is big enough,
    # otherwise sample with replacement). Partition the picks into k
    # groups of m frames each; return the per-group mean. Small `m` keeps
    # inter-group variance high — essential for ap_mr to find a signal to
    # separate on, since fully-random partitioning of the whole stack
    # would just give k copies of the global mean.
    need = k_int * m
    if need <= n:
        idx = rng.choice(n, size=need, replace=False)
    else:
        idx = rng.choice(n, size=need, replace=True)
    idx = idx.reshape(k_int, m)
    out = np.zeros((k_int, ny, nx), dtype=np.float32)
    counts = np.full(k_int, m, dtype=np.int32)
    for c in range(k_int):
        out[c] = s[idx[c]].mean(axis=0)
    return {"refs": d(out), "counts": d(counts)}


register(N(
    "stack.random_group_mean", "Random K-group means", "Stack",
    "stack:stack", "refs:stack,counts:stack",
    [
        Param("k", "int", 4, None, 1, 128),
        Param("samples_per_group", "int", 1, None, 1, 1024),
        Param("seed", "int", 0, None, 0, 2**31 - 1),
    ],
    "Pick `k × samples_per_group` random frames from the input stack and "
    "emit `k` per-group mean images (one per group of `samples_per_group` "
    "picks). Useful as the initial-reference seed for iterative multi-"
    "reference classification. Keep `samples_per_group` small (1–5) so "
    "the k means stay DIFFERENT from each other — fully-random partition "
    "of a big stack would give k near-identical global means and ap_mr "
    "wouldn't have any signal to separate on. Change `seed` to resample.",
    _fn_random_group_mean,
))
