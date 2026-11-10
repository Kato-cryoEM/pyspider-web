"""Built-in node — stack.random_select (Random selection) — Stack.

Pick `n` frames at random from the input stack and return them as a new
stack. Useful for bootstrapping, random subset diagnostics, or seeding a
downstream iterative process with a random slice of a big stack.

The `seed` param is auto-populated by the frontend with the current
timestamp when a fresh node is dragged out, so two newly-placed nodes
produce independent random draws without the user having to think about
it. Users can of course overwrite the seed to lock a specific sample.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param
from backend.nodes import h, d


def _fn_random_select(stack, n: int = 1, seed: int = 0,
                       replace: str = "no"):
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3:
        raise ValueError(
            f"stack.random_select: expected a 3D stack, got shape {s.shape}"
        )
    total, ny, nx = s.shape
    k = max(1, int(n))
    rng = np.random.default_rng(int(seed))
    with_replacement = replace == "yes"
    if k > total and not with_replacement:
        raise ValueError(
            f"stack.random_select: asked for {k} frames but input has only "
            f"{total}. Set replace=yes or lower n."
        )
    idx = rng.choice(total, size=k, replace=with_replacement)
    # Sort so output order mirrors the input ordering rather than the
    # random-draw order — easier to reason about downstream.
    idx = np.sort(idx)
    out = s[idx].copy()
    picked = np.asarray(idx, dtype=np.int32)
    return {"stack": d(out), "indices": d(picked)}


register(N(
    "stack.random_select", "Random selection", "Stack",
    "stack:stack",
    "stack:stack,indices:stack",
    [
        Param("n", "int", 1, None, 1, 100000),
        Param("seed", "int", 0, None, 0, 2**31 - 1),
        Param("replace", "select", "no", ["no", "yes"]),
    ],
    "Pick `n` frames at random from the input stack (without replacement "
    "by default) and emit them as a smaller stack. `indices` also reports "
    "which original frame indices were kept. `seed` is auto-set to the "
    "node's drop-time timestamp, so each freshly-placed node yields an "
    "independent random sample without the user having to think about "
    "seeds; overwrite it to lock a reproducible draw.",
    _fn_random_select,
))
