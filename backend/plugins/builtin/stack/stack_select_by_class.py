"""Built-in node — stack.select_by_class (Filter frames by class id) — Stack.

Given an (aligned) stack and the per-frame assignments produced by a
multi-reference alignment (``align.ap_mr`` → ``assignments_csv``, with
``ref_idx`` as the first column), emit a sub-stack containing only the
frames whose class id equals ``class_id``. Mirrors SPIDER's ``SELECT``
docfile filter: pick rows from a docfile matching a condition, then use
their indices to extract frames.

Pair with ``stack.mean`` (= SPIDER's ``AS R``) to compute one class
average, then ``stack.build`` (2–4) to combine the K per-class means
back into a reference stack for the next iteration.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param
from backend.nodes import h, d


def _fn_select_by_class(stack, assignments_csv: str = "", class_id: int = 0):
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3:
        raise ValueError(
            f"stack.select_by_class: expected a 3D stack, got shape {s.shape}"
        )
    n, ny, nx = s.shape

    rows = [ln for ln in (assignments_csv or "").strip().splitlines() if ln.strip()]
    if len(rows) != n:
        raise ValueError(
            f"stack.select_by_class: assignments_csv has {len(rows)} rows but "
            f"stack has {n} frames — must match 1:1"
        )
    cid = int(class_id)
    keep = np.zeros(n, dtype=bool)
    for i, ln in enumerate(rows):
        body = ln.split("#", 1)[0]
        tok = body.strip().split(",")
        try:
            keep[i] = int(float(tok[0])) == cid
        except (ValueError, IndexError):
            # Malformed row — treat as "no match".
            keep[i] = False
    out = s[keep]
    if out.shape[0] == 0:
        # Empty class: emit a single zero frame rather than a zero-length
        # stack so downstream stack.mean / stack.build still have something
        # to chew on (empty classes happen early in iteration when K-means
        # drops a cluster).
        out = np.zeros((1, ny, nx), dtype=np.float32)
    return {"stack": d(out), "n_matched": int(keep.sum())}


register(N(
    "stack.select_by_class", "Select by class id", "Stack",
    "stack:stack,assignments_csv:string",
    "stack:stack,n_matched:scalar",
    [Param("class_id", "int", 0, None, 0, 128)],
    "Filter a stack to only the frames whose class id matches `class_id` "
    "(first column of each row of `assignments_csv` — the format emitted "
    "by align.ap_mr / ap_md). Mirrors SPIDER's SELECT docfile filter. "
    "Pair with stack.mean (= AS R) to compute one class average; wire K "
    "copies of this + stack.mean + stack.build to produce the K-ref "
    "stack for the next iteration of ap_mr.",
    _fn_select_by_class,
))
