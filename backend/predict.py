"""Predict each node's primary image-output shape WITHOUT running it.

Used by the frontend to size the pre-run preview slot on every node —
so wiring `clock face` (256²) → `polar transform` immediately shows the
correct 128 × ~804 aspect placeholder, and running the graph doesn't
suddenly shove nodes around.

The prediction is per-node, best-effort. Anything with unknown output
shape returns None; the frontend falls back to a tiny stub for those.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Callable

# Prediction functions: (params, input_shapes) -> tuple[int, ...] | None.
# input_shapes is a list-of-shapes matching the node's declared input ports
# in order; a missing wire is represented as None.
_PREDICT: dict[str, Callable[[dict, list], tuple | None]] = {}


def _register(*node_types: str):
    def wrap(fn):
        for t in node_types:
            _PREDICT[t] = fn
        return fn
    return wrap


# ---- Data sources (shape from params only) ------------------------------

@_register("data.clock", "data.camera", "data.astronaut", "data.cell",
           "data.checker", "data.coins", "data.face", "data.res",
           "data.siemens", "data.mo", "data.pinwheel")
def _p_square(params, inputs):
    s = int(params.get("size", 256))
    return (s, s)


@_register("data.betagal")
def _p_betagal(params, inputs):
    v = params.get("variant", "average")
    if v == "map3d":
        s = int(params.get("size", 256))
        if s <= 0:
            s = 256
        return (s, s, s)
    if v == "stack_100":
        return (100, 130, 130)
    # average / raw_particle → single 2D frame ~130² for RELION β-gal
    return (130, 130)


@_register("data.mo3d")
def _p_mo3d(params, inputs):
    nz = int(params.get("nz", 32))
    ny = int(params.get("ny", 32))
    nx = int(params.get("nx", 32))
    return (nz, ny, nx)


# ---- Passthrough shape (same as first input) ----------------------------

@_register("geom.rotate", "geom.shift", "geom.rot_shift", "geom.resize",
           "geom.wi", "arith.abs", "arith.log", "arith.exp",
           "arith.expr", "arith.calc",
           "arith.add", "arith.add2", "arith.sub", "arith.mul", "arith.div",
           "arith.addf", "arith.mulf",
           "filter.gauss", "filter.median", "filter.lp", "filter.hp",
           "filter.bp", "stack.rotate", "stack.shift", "stack.apply_align",
           "align.cc", "align.ccc")
def _p_passthrough(params, inputs):
    # Skip scalar/None inputs — return the FIRST array-shaped upstream so
    # Calculator etc. correctly reports "no preview" when all its inputs
    # are scalars (dx, nx, …).
    for s in (inputs or []):
        if s is not None:
            return s
    return None


# ---- Polar transform ----------------------------------------------------

@_register("geom.po")
def _p_polar(params, inputs):
    if not inputs or inputs[0] is None:
        return None
    shape = inputs[0]
    if len(shape) < 2:
        return None
    ny, nx = int(shape[-2]), int(shape[-1])
    outer_r = int(params.get("outer_radius", 0))
    if outer_r <= 0:
        outer_r = min(ny, nx) // 2
    inner_r = max(0, int(params.get("inner_radius", 1)))
    n_rows = max(1, outer_r - inner_r + 1)
    n_cols = int(round(outer_r * 2.0 * math.pi))
    return (n_rows, n_cols)


# ---- PK: primary preview output is an RGB (H, W, 3) same as input map --

@_register("align.peaks")
def _p_peaks(params, inputs):
    if not inputs or inputs[0] is None:
        return None
    shape = inputs[0]
    if len(shape) < 2:
        return None
    ny, nx = int(shape[-2]), int(shape[-1])
    return (ny, nx, 3)


# ---- Stack ops ----------------------------------------------------------

@_register("stack.ass", "stack.mean", "stack.va", "stack.var")
def _p_stack_reduce(params, inputs):
    """Reduce a stack over its frame axis."""
    if not inputs or inputs[0] is None:
        return None
    s = inputs[0]
    if len(s) >= 3:
        return (int(s[-2]), int(s[-1]))
    return s


@_register("stack.index")
def _p_stack_index(params, inputs):
    if not inputs or inputs[0] is None:
        return None
    s = inputs[0]
    if len(s) >= 3:
        return (int(s[-2]), int(s[-1]))
    return s


@_register("stack.build", "stack.collect")
def _p_stack_build(params, inputs):
    """(N, ny, nx) — unknown N until run, guess a small number."""
    if not inputs or inputs[0] is None:
        return None
    s = inputs[0]
    if len(s) == 2:
        return (10, int(s[0]), int(s[1]))
    return s


# ---- Fourier ------------------------------------------------------------

@_register("ft.forward", "ft.inverse")
def _p_ft(params, inputs):
    return inputs[0] if inputs else None


# =========================================================================


def _port_kind(node_type: str, port_name: str, side: str) -> str | None:
    """Look up a port's kind from the live catalog. `side` = 'in' or 'out'."""
    try:
        from .nodes import CATALOG
    except Exception:
        return None
    spec = CATALOG.get(node_type)
    if spec is None:
        return None
    ports = spec.outputs if side == "out" else spec.inputs
    for p in ports:
        if p.name == port_name:
            return p.kind
    return None


def predict_shapes(graph: dict) -> dict[str, list[int]]:
    """Return {node_id: [shape...]} for each node whose primary image
    output shape can be predicted. Nodes with unknown / non-image output
    are omitted (frontend then shows a compact stub).
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Topological order over the whole graph (parent info is irrelevant for
    # shape propagation — a nested-DO child still gets the same shape as
    # its upstream feed).
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    ids = [n["id"] for n in nodes]
    for e in edges:
        s, d = e.get("src") or e.get("source"), e.get("dst") or e.get("target")
        if s in by_id and d in by_id:
            incoming[d].add(s)
            outgoing[s].add(d)
    ready = deque(nid for nid in ids if not incoming[nid])
    order: list[str] = []
    remaining = {nid: set(incoming[nid]) for nid in ids}
    while ready:
        nid = ready.popleft()
        order.append(nid)
        for succ in outgoing[nid]:
            remaining[succ].discard(nid)
            if not remaining[succ]:
                ready.append(succ)

    # For each edge, record what shape it delivers.
    per_edge_shape: dict[tuple[str, str, str], tuple | None] = {}
    # But we don't have per-port shape mapping for outputs — we assume the
    # node's predicted shape represents its PRIMARY image output. This
    # matches ~all common flows (rotate/PO/CC/PK/etc.) since these emit a
    # single image; multi-output nodes fall through cleanly.

    result: dict[str, list[int]] = {}
    node_shape: dict[str, tuple | None] = {}

    for nid in order:
        node = by_id.get(nid) or {}
        typ = node.get("type", "")
        params = node.get("params", {}) or {}

        # Gather input shapes — but ONLY for edges whose source port emits
        # a shape-bearing kind (image/stack/volume/fourier). Scalar/string
        # sources don't carry a spatial shape, so propagating the upstream
        # node's primary output shape (e.g. PK's preview 3D) as calc's
        # input would be wrong.
        inp_shapes: list[tuple | None] = []
        for e in edges:
            s = e.get("src") or e.get("source")
            d = e.get("dst") or e.get("target")
            sp = e.get("srcPort") or e.get("sourceHandle") or ""
            if d != nid or s not in node_shape:
                continue
            src_type = (by_id.get(s) or {}).get("type", "")
            k = _port_kind(src_type, sp, "out")
            if k in ("image", "stack", "volume", "fourier", None):
                inp_shapes.append(node_shape[s])
            else:
                inp_shapes.append(None)   # scalar / string → no shape

        fn = _PREDICT.get(typ)
        shape = fn(params, inp_shapes) if fn else None
        node_shape[nid] = shape
        if shape is not None:
            result[nid] = list(shape)

    return result
