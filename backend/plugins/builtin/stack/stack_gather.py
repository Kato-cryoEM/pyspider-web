"""Built-in node — stack.gather (Pick frames by index list) — Stack.

Given a reference stack and a 1D list of integer indices, emit a new
stack where each output frame is ``references[indices[i]]`` — i.e. the
reference image that the alignment chose for that particle.

Typical use: wire ``align.ap_m.ref_index`` (shape ``(n_particles,)``)
and the original ``recon.pj3q.stack`` of projections. The result is an
``n_particles``-frame stack where each frame is the matched reference,
side-by-side-able with ``align.ap_m.aligned`` to visually confirm which
reference each particle got assigned to.

Also accepts a plain comma / whitespace separated ``indices_csv`` string
(wireable; falls back to the ``indices`` stack port) so the first
column of a SPIDER-style ``assignments_csv`` can be fed in directly
without an extra extraction node.
"""

from __future__ import annotations

import re
import numpy as np

from backend.plugin_api import register, N, Param, Port
from backend.nodes import h, d


def _parse_indices(indices_csv: str) -> list[int]:
    """Pull integer indices out of a doc-file style string. Accepts one
    index per line (first token) or a bare comma/space separated list."""
    out: list[int] = []
    s = (indices_csv or "").strip()
    if not s:
        return out
    # Doc-file rows with # comments, where the first column is ref_idx.
    for row in s.splitlines():
        row = row.split("#", 1)[0].strip()
        if not row:
            continue
        tok = re.split(r"[,\s]+", row)[0]
        try:
            out.append(int(float(tok)))
        except ValueError:
            continue
    return out


def _fn_stack_gather(references, indices=None, indices_csv: str = ""):
    refs = h(references).astype(np.float32)
    if refs.ndim == 2:
        refs = refs[None, ...]
    if refs.ndim != 3:
        raise ValueError(
            f"stack.gather: references must be a 3D stack, got {refs.shape}"
        )
    n_ref = int(refs.shape[0])

    # Pull indices from either the `indices` port (preferred: numeric array
    # straight off ap_m.ref_index / stack.random_select.indices) or the
    # text param (first column of an assignments_csv).
    if indices is not None:
        idx = h(indices)
        idx = np.asarray(idx).reshape(-1).astype(np.int64, copy=False)
    else:
        parsed = _parse_indices(indices_csv)
        if not parsed:
            raise ValueError(
                "stack.gather: wire the `indices` port, or fill "
                "`indices_csv` with one integer per row."
            )
        idx = np.asarray(parsed, dtype=np.int64)

    # Guard against out-of-range indices so a mis-wired graph fails with a
    # clear message instead of a numpy IndexError.
    bad = (idx < 0) | (idx >= n_ref)
    if bad.any():
        offenders = np.where(bad)[0][:5].tolist()
        raise ValueError(
            f"stack.gather: {int(bad.sum())} index values are outside "
            f"[0, {n_ref - 1}] (e.g. positions {offenders}). Did the "
            "reference stack and index source come from the same alignment?"
        )
    out = refs[idx].copy()
    return {"stack": d(out.astype(np.float32)),
            "n_frames": int(len(idx))}


register(N(
    "stack.gather", "Select refs by index", "Stack",
    [Port("references", "stack"),
     Port("indices",    "stack", optional=True)],
    "stack:stack,n_frames:scalar",
    [Param("indices_csv", "string", "", wireable=True,
           placeholder="one int per row (or wire indices / assignments_csv)")],
    "Build a stack whose i-th frame is `references[indices[i]]`. Typical "
    "wiring: `align.ap_m.ref_index` → `indices` and `recon.pj3q.stack` → "
    "`references` to emit the per-particle matched reference as a side-by-"
    "side stack with `align.ap_m.aligned`. `indices_csv` is an alternative "
    "text input that accepts one integer per row (first column of a "
    "SPIDER-style `assignments_csv` is parsed correctly), used when "
    "`indices` is unwired. Out-of-range indices raise a clear error so a "
    "misrouted pipeline doesn't produce silent garbage.",
    _fn_stack_gather,
))
