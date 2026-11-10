"""DAG execution engine for the node graph.

Graph JSON shape (from the frontend):

    {
      "nodes":  [ {"id": "n1", "type": "filter.gauss_lp", "params": {"fc": 0.1},
                   "parent": null | "<do_node_id>"}, ... ],
      "edges":  [ {"src": "n1", "srcPort": "image",
                   "dst": "n2", "dstPort": "image"}, ... ]
    }

Execution rule:
  - Nodes with a `parent` set to a `flow.do` container are the DO's children.
    They do NOT run in the top-level topological order — instead the DO node
    runs them repeatedly, once per loop iteration.
  - Top-level nodes run in Kahn topological order over the edges among them.
  - When a `flow.do` node is reached, the engine iterates its child subgraph
    from `start` to `end` (inclusive) with the given `step`, seeding the DO's
    own `i` output each pass so children can consume it via a param wire.
  - After the loop, each child holds its final-iteration NodeResult and the
    DO node itself reports the last value of `i`.

Error policy: on failure the offending node's error is captured and returned
in the result; downstream nodes are skipped. Inside a DO loop, a failure in
one child does not abort remaining iterations — the last iteration's error
survives to the returned result.
"""

from __future__ import annotations

import queue
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .nodes import CATALOG, NodeSpec
from . import progress as _progress

# WASM builds of CPython (Pyodide / emscripten) lack real thread support, so
# `threading.Thread.start()` raises RuntimeError. When we detect that we run
# the node body inline on the caller's thread.
_NO_THREADS = sys.platform in ("emscripten", "wasi")


@dataclass
class NodeResult:
    id: str
    ok: bool
    outputs: dict[str, Any]        # port name -> value
    error: str | None = None


def _topo_order(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Kahn's algorithm.  Raises ValueError on cycles."""
    ids = [n["id"] for n in nodes]
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    id_set = set(ids)
    for e in edges:
        if e["src"] in id_set and e["dst"] in id_set:
            incoming[e["dst"]].add(e["src"])
            outgoing[e["src"]].add(e["dst"])

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

    if len(order) != len(ids):
        raise ValueError("graph has a cycle")
    return order


def _build_wire(edges: list[dict]) -> dict[tuple[str, str], tuple[str, str]]:
    """Map (dst_id, dst_port) -> (src_id, src_port)."""
    wire: dict[tuple[str, str], tuple[str, str]] = {}
    for e in edges:
        wire[(e["dst"], e["dstPort"])] = (e["src"], e["srcPort"])
    return wire


def _do_range(start: int, end: int, step: int) -> Iterator[int]:
    """SPIDER DO semantics: inclusive of `end`."""
    if step == 0:
        return
    if step > 0:
        i = start
        while i <= end:
            yield i
            i += step
    else:
        i = start
        while i >= end:
            yield i
            i += step


def _execute_node_streamed(node: dict,
                           wire: dict[tuple[str, str], tuple[str, str]],
                           results: dict[str, NodeResult],
                           by_id: dict[str, dict]):
    """Run one node in a worker thread while pumping ``report_progress`` calls
    back to the caller as ``node_progress`` events. Blocks until the worker
    completes, then the caller yields ``node_done`` as before.

    The worker installs a per-context emitter that pushes ``(progress, log)``
    tuples onto a queue; this generator drains the queue between polls of
    the worker thread. A sentinel ``None`` is enqueued on completion so we
    know when to stop.
    """
    nid = node["id"]
    q: "queue.Queue[tuple[float | None, str | None] | None]" = queue.Queue()

    def emitter(prog: float | None, log: str | None) -> None:
        q.put((prog, log))

    if _NO_THREADS:
        token = _progress.set_emitter(emitter)
        try:
            _execute_node(node, wire, results, by_id)
        finally:
            _progress.reset_emitter(token)
        while not q.empty():
            item = q.get_nowait()
            if item is None:
                break
            yield ("node_progress", nid, item[0], item[1])
        return

    def worker() -> None:
        token = _progress.set_emitter(emitter)
        try:
            _execute_node(node, wire, results, by_id)
        finally:
            _progress.reset_emitter(token)
            q.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            item = q.get(timeout=0.05)
        except queue.Empty:
            if not t.is_alive():
                # Race: thread died without enqueuing sentinel. Drain remainder.
                while not q.empty():
                    left = q.get_nowait()
                    if left is None:
                        break
                    yield ("node_progress", nid, left[0], left[1])
                break
            continue
        if item is None:
            break
        prog, log = item
        yield ("node_progress", nid, prog, log)
    t.join(timeout=1.0)


def _src_port_kind(src_id: str, src_port: str,
                   by_id: dict[str, dict]) -> str | None:
    """Look up the port kind declared by the upstream node's spec, so the
    engine can distinguish `stack:stack` (broadcast per frame) from
    `volume:volume` (auto-take middle slice) when both feed an `image` port."""
    src_node = by_id.get(src_id)
    if not src_node:
        return None
    src_spec = CATALOG.get(src_node.get("type"))
    if not src_spec:
        return None
    for op in src_spec.outputs:
        if op.name == src_port:
            return op.kind
    return None


def _execute_node(node: dict,
                  wire: dict[tuple[str, str], tuple[str, str]],
                  results: dict[str, NodeResult],
                  by_id: dict[str, dict] | None = None) -> None:
    """Run a single non-container node, mutating `results` in place."""
    nid = node["id"]
    spec: NodeSpec | None = CATALOG.get(node["type"])
    # Special case: stack.collect used OUTSIDE a DO just passes through as
    # a length-1 stack. Inside a DO it's handled by _stream_do_loop instead.
    if node.get("type") == "stack.collect":
        src = wire.get((nid, "image"))
        if src is None:
            results[nid] = NodeResult(nid, ok=False, outputs={},
                                      error="stack.collect: `image` not wired")
            return
        src_id, src_port = src
        up = results.get(src_id)
        if up is None or not up.ok or src_port not in up.outputs:
            results[nid] = NodeResult(nid, ok=False, outputs={},
                                      error="stack.collect: upstream failed")
            return
        arr = np.asarray(up.outputs[src_port], dtype=np.float32)
        results[nid] = NodeResult(nid, ok=True,
                                  outputs={"stack": arr[np.newaxis, ...]})
        return
    if spec is None or spec.fn is None:
        results[nid] = NodeResult(nid, ok=False, outputs={},
                                  error=f"unknown node type {node['type']!r}")
        return

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for port in spec.inputs:
        src = wire.get((nid, port.name))
        if src is None:
            if not port.optional:
                missing.append(port.name)
            continue
        src_id, src_port = src
        up = results.get(src_id)
        if up is None or not up.ok or src_port not in up.outputs:
            if not port.optional:
                missing.append(port.name)
            continue
        val = up.outputs[src_port]
        # Auto-unwrap 3D arrays feeding an `image` port ONLY when the source
        # is a `volume` — a representative middle slice is nearly always what
        # the user wants when peeking at a 3D density map.
        # A `stack` source is passed through intact so that per-frame
        # broadcasting works: e.g. a PJ 3Q batch (N, ny, nx) × a TF C CTF
        # (ny, nx) in MU yields per-frame CTF-weighted projections without
        # any Stack · wrapper. Insert `stack.index` if you want one frame.
        # Arithmetic / filter / mask nodes are also exempted so a scalar
        # multiply or element-wise op preserves 3D volumes (e.g. SGD V
        # update: V ← V + lr · grad on 128³ volumes).
        node_type = node.get("type", "")
        # arith / filter / mask broadcast element-wise on 3D volumes.
        # ft handles the 2D-vs-3D dispatch itself (dims=2d/3d param), so it
        # must receive the whole volume, not a middle slice.
        _VOLUME_PRESERVING = ("arith.", "filter.", "mask.", "ft.")
        if (port.kind == "image"
                and hasattr(val, "ndim") and val.ndim == 3
                and not node_type.startswith(_VOLUME_PRESERVING)):
            src_kind = _src_port_kind(src_id, src_port, by_id or {})
            if src_kind == "volume":
                val = val[val.shape[0] // 2]
        kwargs[port.name] = val

    if missing:
        results[nid] = NodeResult(nid, ok=False, outputs={},
                                  error=f"missing input(s): {', '.join(missing)}")
        return

    for p in spec.params:
        src = wire.get((nid, p.name))
        if src is not None:
            src_id, src_port = src
            up = results.get(src_id)
            if up and up.ok and src_port in up.outputs:
                kwargs[p.name] = up.outputs[src_port]
                continue
        if p.name in node.get("params", {}):
            kwargs[p.name] = node["params"][p.name]
        else:
            kwargs[p.name] = p.default

    try:
        out = spec.fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        results[nid] = NodeResult(nid, ok=False, outputs={}, error=repr(exc))
        return

    if not isinstance(out, dict):
        results[nid] = NodeResult(nid, ok=False, outputs={},
                                  error=f"node returned non-dict: {type(out).__name__}")
        return

    results[nid] = NodeResult(nid, ok=True, outputs=out)


def _resolve_do_param(do_node: dict, name: str, default: int,
                      wire: dict[tuple[str, str], tuple[str, str]],
                      results: dict[str, NodeResult]) -> int:
    """A DO param may be wired (e.g. n_rows → end). Look up the wire first,
    then fall back to the hand-typed value, then to the schema default."""
    src = wire.get((do_node["id"], name))
    if src is not None:
        src_id, src_port = src
        up = results.get(src_id)
        if up and up.ok and src_port in up.outputs:
            try:
                return int(up.outputs[src_port])
            except (TypeError, ValueError):
                pass
    v = do_node.get("params", {}).get(name, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def run(graph: dict) -> dict[str, NodeResult]:
    """Execute the graph.  Returns per-node results."""
    results: dict[str, NodeResult] = {}
    for evt in run_stream(graph):
        if evt[0] == "done":
            results = evt[1]
    return results


def run_stream(graph: dict):
    """Streaming version of `run` — yields per-node events as tuples:

        ("node_start", nid)                 emitted just before a node runs
        ("node_done",  nid, NodeResult)     emitted right after it finishes
        ("done", results_dict)              final event with the full mapping

    Inside a DO container, children emit start/done on each iteration so the
    UI can flash them in sync with the loop.
    """
    nodes: list[dict] = graph.get("nodes", [])
    edges: list[dict] = graph.get("edges", [])
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # `children_of[do_id]` = all nodes whose immediate parent is that DO.
    # Also true for nested DOs — an inner DO's own children go into its own
    # bucket, not the outer's, so `_stream_do_loop` can recurse cleanly.
    children_of: dict[str, list[dict]] = defaultdict(list)
    top_nodes: list[dict] = []
    for n in nodes:
        p = n.get("parent")
        if p and p in by_id and by_id[p].get("type") == "flow.do":
            children_of[p].append(n)
        else:
            top_nodes.append(n)

    top_ids = {n["id"] for n in top_nodes}

    # For outer topo we need to project every edge onto its OUTERMOST
    # endpoints so cross-boundary wires still impose the right ordering.
    # Example: (top-level AS S) → (state inside DO_X) becomes AS S → DO_X
    # in the outer topo, so AS S runs before DO_X iterates. Without this
    # projection DO_X had no incoming outer edges, could be sorted before
    # AS S, and state.initial would look up AS S's result before it existed.
    def _outer_id(nid: str) -> str | None:
        n = by_id.get(nid)
        if not n:
            return None
        if nid in top_ids:
            return nid
        p = n.get("parent")
        # Recurse in case of future nested DOs; today parent is at most one
        # level deep so this bottoms out quickly.
        return _outer_id(p) if p else None

    top_edges: list[dict] = []
    seen_top_edge: set[tuple[str, str]] = set()
    for e in edges:
        so = _outer_id(e["src"])
        do_ = _outer_id(e["dst"])
        if so is None or do_ is None or so == do_:
            # so == do_ happens for wires entirely inside one DO — those are
            # handled by _stream_do_loop, not the outer topo.
            continue
        key = (so, do_)
        if key in seen_top_edge:
            continue
        seen_top_edge.add(key)
        top_edges.append({"src": so, "dst": do_})
    order = _topo_order(top_nodes, top_edges)

    wire = _build_wire(edges)
    results: dict[str, NodeResult] = {}

    for nid in order:
        node = by_id[nid]
        yield ("node_start", nid)
        if node.get("type") == "flow.do":
            yield from _stream_do_loop(node, children_of[nid], wire, results,
                                        by_id, children_of)
        else:
            yield from _execute_node_streamed(node, wire, results, by_id)
        yield ("node_done", nid, results.get(nid))

    yield ("done", results)


def _stream_do_loop(do_node, children, wire, results, by_id,
                    children_of=None):
    """Same as _run_do_loop but yields per-iteration events for each child."""
    nid = do_node["id"]
    start = _resolve_do_param(do_node, "start", 1, wire, results)
    end   = _resolve_do_param(do_node, "end",  10, wire, results)
    step  = _resolve_do_param(do_node, "step",  1, wire, results)

    child_ids = [c["id"] for c in children]
    child_id_set = set(child_ids)

    # flow.state nodes hold a value that changes across iterations. The wire
    # feeding value_in comes from a DOWNSTREAM node, which would create a
    # cycle in the intra-DO DAG — so we filter it out of the topo sort and
    # handle capture manually at end of each iteration.
    state_ids = {c["id"] for c in children
                 if by_id.get(c["id"], {}).get("type") == "flow.state"}

    # Project each edge onto the child_id_set so nested-DO endpoints
    # participate in this DO's topo through their outer parent (the
    # inner DO). Analogous to _outer_id at the top level.
    def _child_ancestor(node_id: str) -> str | None:
        cur = node_id
        while cur:
            if cur in child_id_set:
                return cur
            cur = (by_id.get(cur) or {}).get("parent")
        return None

    inner_edges: list[dict] = []
    seen_pair: set[tuple[str, str]] = set()
    for (dst, dp), (src, sp) in wire.items():
        sa = _child_ancestor(src)
        da = _child_ancestor(dst)
        if sa is None or da is None or sa == da:
            continue
        if da in state_ids and dp == "value_in" and dst == da:
            # Feedback edge to a direct state child — filter as before so
            # the DAG stays acyclic within one iteration.
            continue
        key = (sa, da)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        # Keep original src/dst/port only when both are direct children;
        # otherwise the "edge" is a synthesized DAG dependency only.
        inner_edges.append({"src": sa, "srcPort": sp if sa == src else "",
                            "dst": da, "dstPort": dp if da == dst else ""})
    try:
        order = _topo_order(children, inner_edges)
    except ValueError as exc:
        results[nid] = NodeResult(nid, ok=False, outputs={},
                                  error=f"cycle inside DO subgraph: {exc}")
        return

    # Each stack.collect child gets both an append-order list AND an
    # index-keyed dict. If the `index` input is wired, the dict is used
    # (= imperative `stack[i] = image`); otherwise frames are consumed
    # in append order at end-of-loop.
    collectors = {cid: [] for cid in child_ids
                  if by_id[cid].get("type") == "stack.collect"}
    collectors_indexed: dict[str, dict[int, Any]] = {
        cid: {} for cid in collectors
    }
    collectors_use_index: dict[str, bool] = {cid: False for cid in collectors}

    # Seed each flow.state from its `initial` upstream (which lives outside
    # the DO container and has already run in the outer topo order).
    state_vals: dict[str, Any] = {}
    state_seed_errors: dict[str, str] = {}
    for sid in state_ids:
        src = wire.get((sid, "initial"))
        if src is None:
            # Report each incoming edge to this state node so users can spot
            # the mistake (e.g. wire went to `value_in` instead of `initial`).
            incoming = [dp for (dst, dp) in wire.keys() if dst == sid]
            if incoming:
                state_seed_errors[sid] = (
                    f"flow.state: no wire on `initial`. Found wires on: "
                    f"{', '.join(incoming)}. Wire your seed value to "
                    f"`initial`, not `{incoming[0]}`.")
            else:
                state_seed_errors[sid] = (
                    "flow.state: no wire on `initial`. Connect an image "
                    "from OUTSIDE the DO Loop to the state node's `initial` "
                    "input (top handle) as the iteration-1 seed.")
            continue
        src_id, src_port = src
        up = results.get(src_id)
        if up is None:
            # Diagnose why the upstream didn't run. The usual cause is that
            # the upstream node was accidentally dropped INSIDE the DO
            # container, so it's a child of DO instead of a top-level node
            # that runs once before iterations start.
            upstream = by_id.get(src_id, {})
            up_type = upstream.get("type", "?")
            up_parent = upstream.get("parent")
            if up_parent == nid:
                state_seed_errors[sid] = (
                    f"flow.state: `initial` upstream ({up_type}) is INSIDE "
                    f"this DO Loop — but it must run BEFORE iterations "
                    f"start to seed the state. Move it OUTSIDE the red DO "
                    f"frame (drag it out) and re-run.")
            elif up_parent:
                state_seed_errors[sid] = (
                    f"flow.state: `initial` upstream ({up_type}) is inside "
                    f"a DIFFERENT DO container `{up_parent}`, which hadn't "
                    f"executed by the time this DO started. Move the "
                    f"upstream outside all DO containers.")
            else:
                state_seed_errors[sid] = (
                    f"flow.state: `initial` upstream ({up_type}, id={src_id}) "
                    f"was never executed. Check for a cycle in the outer "
                    f"graph or a duplicate node.")
        elif not up.ok:
            state_seed_errors[sid] = (
                f"flow.state: `initial` upstream failed — {up.error}")
        elif src_port not in up.outputs:
            state_seed_errors[sid] = (
                f"flow.state: `initial` upstream produced no `{src_port}` "
                f"output (has: {list(up.outputs.keys())})")
        else:
            state_vals[sid] = up.outputs[src_port]

    i_values = list(_do_range(start, end, step))
    if not i_values:
        results[nid] = NodeResult(nid, ok=True, outputs={"i": start})
        for cid in collectors:
            results[cid] = NodeResult(cid, ok=True,
                                      outputs={"stack": np.zeros((0, 1, 1), np.float32)})
        return

    last_i = start
    total_iters = len(i_values)
    for iter_idx, i in enumerate(i_values):
        last_i = i
        results[nid] = NodeResult(nid, ok=True, outputs={"i": i})
        # Surface the current iteration on the DO node itself so the UI can
        # show "iter k / N" without having to peek at per-child events.
        # Emitted as both progress (for a bar) and a log line (for the
        # Inspector tail).
        yield ("node_progress", nid,
               (iter_idx + 1) / max(1, total_iters),
               f"iter {iter_idx + 1} / {total_iters}  (i = {i})")
        # Publish current state values as value_out for this iteration —
        # done before children run so downstream sees them.
        for sid in state_ids:
            if sid in state_vals:
                results[sid] = NodeResult(
                    sid, ok=True, outputs={"value_out": state_vals[sid]})
            else:
                results[sid] = NodeResult(
                    sid, ok=False, outputs={},
                    error=state_seed_errors.get(
                        sid,
                        "flow.state: no initial value (wire `initial`)"))
        for cid in order:
            yield ("node_start", cid)
            if cid in state_ids:
                # value_out was already populated above; nothing more to do.
                pass
            elif cid in collectors:
                src = wire.get((cid, "image"))
                if src is None:
                    results[cid] = NodeResult(cid, ok=False, outputs={},
                                              error="stack.collect: `image` not wired")
                else:
                    src_id, src_port = src
                    up = results.get(src_id)
                    if up is None or not up.ok or src_port not in up.outputs:
                        results[cid] = NodeResult(cid, ok=False, outputs={},
                                                  error="stack.collect: upstream failed")
                    else:
                        arr = np.asarray(up.outputs[src_port], dtype=np.float32)
                        # Resolve optional `index` wire (scalar). If wired,
                        # imperative mode: output_stack[index] = image.
                        idx_src = wire.get((cid, "index"))
                        idx_val = None
                        if idx_src is not None:
                            isid, isp = idx_src
                            iup = results.get(isid)
                            if iup and iup.ok and isp in iup.outputs:
                                try:
                                    idx_val = int(iup.outputs[isp])
                                except (TypeError, ValueError):
                                    idx_val = None
                        if idx_val is not None and idx_val >= 0:
                            collectors_use_index[cid] = True
                            collectors_indexed[cid][idx_val] = arr
                            # Build a dense preview up to the current max index.
                            max_i = max(collectors_indexed[cid].keys())
                            frames_so_far = []
                            for k in range(max_i + 1):
                                frames_so_far.append(
                                    collectors_indexed[cid].get(
                                        k, np.zeros_like(arr)))
                            results[cid] = NodeResult(
                                cid, ok=True,
                                outputs={"stack": np.stack(frames_so_far, axis=0)})
                        else:
                            collectors[cid].append(arr)
                            results[cid] = NodeResult(
                                cid, ok=True,
                                outputs={"stack": np.stack(collectors[cid], axis=0)})
            elif by_id[cid].get("type") == "flow.do":
                # Nested DO — recurse. Uses the same children_of map so its
                # own children are found. Its full inner loop runs to
                # completion inside a single iteration of the OUTER loop.
                inner_kids = (children_of or {}).get(cid, [])
                yield from _stream_do_loop(by_id[cid], inner_kids, wire,
                                            results, by_id, children_of)
            else:
                _execute_node(by_id[cid], wire, results, by_id)
            yield ("node_done", cid, results.get(cid))
        # Capture each flow.state's value_in NOW (end of iteration k) so it
        # becomes the value_out at the start of iteration k+1.
        for sid in state_ids:
            src = wire.get((sid, "value_in"))
            if src is None:
                continue
            src_id, src_port = src
            up = results.get(src_id)
            if up and up.ok and src_port in up.outputs:
                state_vals[sid] = up.outputs[src_port]

    results[nid] = NodeResult(nid, ok=True, outputs={"i": last_i})
    # Finalise stack.collect outputs. Indexed-mode collectors assemble the
    # dense output stack now that every slot has been seen; empty collectors
    # fall back to a zero-length 3D stack so downstream type checks pass.
    for cid in collectors:
        if collectors_use_index.get(cid) and collectors_indexed[cid]:
            idx_map = collectors_indexed[cid]
            max_i = max(idx_map.keys())
            template = next(iter(idx_map.values()))
            frames = [idx_map.get(k, np.zeros_like(template))
                      for k in range(max_i + 1)]
            results[cid] = NodeResult(cid, ok=True,
                                      outputs={"stack": np.stack(frames, axis=0)})
            continue
        frames = collectors[cid]
        if not frames:
            results[cid] = NodeResult(cid, ok=True,
                                      outputs={"stack": np.zeros((0, 1, 1), np.float32)})
