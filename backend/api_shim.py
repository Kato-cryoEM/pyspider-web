"""In-browser (Pyodide) API shim.

Mirrors the FastAPI endpoints exposed by `backend/main.py` so the frontend
can call the same operations via a Pyodide runtime instead of HTTP. Each
function takes/returns JSON-serialisable Python objects (no numpy arrays,
image buffers encoded as base64).

All functions accept JSON strings for structured input to keep the Pyodide
boundary simple (strings marshal trivially through pyodide.runPythonAsync).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import challenge_sessions, engine, formats, plugin_loader, xp
from .nodes import CATALOG, public_catalog
from .predict import predict_shapes as _predict_shapes
from .preview import encode_output


# `backend/api_shim.py` → `backend/` → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUILTIN_TUTORIAL_DIR = _PROJECT_ROOT / "backend" / "tutorials"

# In the browser, "settings" are purely in-memory — the frontend persists
# them to localStorage if it needs durability. The default save dir is
# meaningless in the browser sandbox but kept so the type stays compatible.
_SETTINGS: dict[str, Any] = {"default_save_dir": ""}


# -- Discovery -------------------------------------------------------------

def get_catalog_json() -> str:
    """Catalog for the Palette — same shape as the old `/api/nodes`."""
    return json.dumps({"nodes": public_catalog()})


def get_plugins_json() -> str:
    """Plugin load log + format handler coverage."""
    return json.dumps({
        "plugins": plugin_loader.load_log(),
        "readers": formats.readers(),
        "writers": formats.writers(),
    })


def get_health_json() -> str:
    return json.dumps({
        "backend": xp.current(),
        "cuda_available": xp.has_cuda(),
    })


# -- Shape prediction ------------------------------------------------------

def predict_shapes_json(req_json: str) -> str:
    req = json.loads(req_json)
    shapes = _predict_shapes({
        "nodes": req.get("nodes", []),
        "edges": req.get("edges", []),
    })
    return json.dumps({"shapes": shapes})


# -- Tutorials -------------------------------------------------------------

def list_tutorials_json() -> str:
    lessons: list[dict] = []
    errors: list[dict] = []
    seen_ids: set[str] = set()
    if _BUILTIN_TUTORIAL_DIR.is_dir():
        for f in sorted(_BUILTIN_TUTORIAL_DIR.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                errors.append({"file": str(f), "error": str(e)})
                continue
            if not (isinstance(obj, dict)
                    and isinstance(obj.get("id"), str)
                    and isinstance(obj.get("title"), str)
                    and isinstance(obj.get("steps"), list)):
                errors.append({"file": str(f),
                               "error": "missing id / title / steps"})
                continue
            if obj["id"] in seen_ids:
                continue
            seen_ids.add(obj["id"])
            lessons.append(obj)
    return json.dumps({"lessons": lessons, "errors": errors})


# -- Challenges ------------------------------------------------------------

def list_challenges_json() -> str:
    specs, errors = challenge_sessions.load_challenges()
    public = [
        {k: v for k, v in s.items()
         if k not in ("compare_to", "metric", "pass_score")}
        for s in specs
    ]
    return json.dumps({"challenges": public, "errors": errors})


def start_challenge_json(req_json: str) -> str:
    req = json.loads(req_json)
    spec = challenge_sessions.get_spec(req["challenge_id"])
    if spec is None:
        return json.dumps({"error": f"challenge {req['challenge_id']!r} not found"})
    try:
        return json.dumps(challenge_sessions.start_session(spec))
    except ValueError as e:
        return json.dumps({"error": str(e)})


def fixture_preview_json(req_json: str) -> str:
    req = json.loads(req_json)
    img = challenge_sessions.get_fixture(req["session_id"], req["name"])
    if img is None:
        return json.dumps({"error": f"no fixture {req['name']!r}"})
    nd = getattr(img, "ndim", 0)
    if nd < 2:
        return json.dumps({"png": {
            "kind": "scalar",
            "shape": list(getattr(img, "shape", ())),
            "dtype": str(getattr(img, "dtype", "")),
            "png": None, "min": None, "max": None,
        }})
    kind = "volume" if nd == 3 else "image"
    return json.dumps({"png": encode_output(img, kind=kind, port_name=req["name"])})


def judge_challenge_json(req_json: str) -> str:
    req = json.loads(req_json)
    spec = challenge_sessions.get_spec(req["challenge_id"])
    if spec is None:
        return json.dumps({"error": f"challenge {req['challenge_id']!r} not found"})
    # Judge-only fixtures: reject graphs that read them directly.
    judge_only = {f["name"] for f in spec.get("fixtures", []) if f.get("judge_only")}
    if judge_only:
        locked_sources: set[str] = set()
        for n in req["nodes"]:
            if n.get("type") not in ("data.challenge_input",
                                     "data.challenge_input_volume"):
                continue
            name = (n.get("params") or {}).get("name")
            if name in judge_only:
                locked_sources.add(n.get("id"))
        if locked_sources:
            for e in req["edges"]:
                if e.get("src") in locked_sources:
                    return json.dumps({"error":
                        "challenge rule: fixture is locked (judge only)"})
    try:
        results = engine.run({"nodes": req["nodes"], "edges": req["edges"]})
    except ValueError as e:
        return json.dumps({"error": f"graph error: {e}"})
    res = results.get(req["answer_node_id"])
    if res is None or not res.ok:
        err = res.error if res else "answer node did not run"
        return json.dumps({"ok": False, "pass": False, "score": None,
                           "error": f"answer node failed: {err}"})
    port = spec.get("answer_port", "image")
    aliases = {"image":  ["image", "value_out", "stack", "volume", "cc",
                           "reference", "aligned", "result", "out"],
               "stack":  ["stack", "class_means", "aligned", "value_out", "image"],
               "volume": ["volume", "stack"],
               "labels": ["ref_index", "labels", "class_ids", "assignments"]}
    picked_port = None
    for alt in aliases.get(port, [port]):
        if alt in res.outputs:
            picked_port = alt
            break
    if picked_port is None:
        for name, v in res.outputs.items():
            if not hasattr(v, "shape"):
                continue
            nd = getattr(v, "ndim", 0)
            if port == "image" and nd in (2, 3):
                picked_port = name; break
            if port == "stack" and nd == 3:
                picked_port = name; break
            if port == "labels" and nd == 1:
                picked_port = name; break
    if picked_port is None:
        return json.dumps({"ok": False, "pass": False, "score": None,
                           "error": f"answer node has no `{port}` output "
                                    f"(has: {sorted(res.outputs.keys())})"})
    val = np.asarray(res.outputs[picked_port])
    if port == "image" and val.ndim == 3:
        val = val[val.shape[0] // 2]
    if port == "labels":
        val = val.astype(int)
    return json.dumps(challenge_sessions.judge(spec, req["session_id"], val))


# -- Settings --------------------------------------------------------------

def get_settings_json() -> str:
    return json.dumps(dict(_SETTINGS))


def update_settings_json(patch_json: str) -> str:
    patch = json.loads(patch_json)
    for k, v in patch.items():
        if v is not None and k in _SETTINGS:
            _SETTINGS[k] = v
    return json.dumps(dict(_SETTINGS))


# -- Graph execution -------------------------------------------------------

def _encode_node_result(nid: str, res, port_kinds: dict[str, str]) -> dict:
    if res is None:
        return {"ok": False, "error": "no result", "outputs": {}}
    out: dict[str, dict] = {}
    for port_name, value in res.outputs.items():
        out[port_name] = encode_output(
            value, kind=port_kinds.get(port_name), port_name=port_name)
    return {"ok": res.ok, "error": res.error, "outputs": out}


def run_graph_json(req_json: str) -> str:
    req = json.loads(req_json)
    try:
        results = engine.run({"nodes": req["nodes"], "edges": req["edges"]})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    node_type_by_id = {n["id"]: n.get("type", "") for n in req["nodes"]}
    payload: dict[str, dict] = {}
    for nid, res in results.items():
        spec = CATALOG.get(node_type_by_id.get(nid, ""))
        port_kinds = {p.name: p.kind for p in spec.outputs} if spec else {}
        payload[nid] = _encode_node_result(nid, res, port_kinds)
    return json.dumps({"results": payload})


_ABORT = {"flag": False}


def request_abort() -> None:
    """Called from the worker on an abort message. The running stream
    checks this between events and bails out at the next safe point
    (end-of-node granularity, since a single node call is synchronous)."""
    _ABORT["flag"] = True


async def run_graph_stream(req_json: str, on_event) -> None:
    """Streaming graph run. Calls `on_event(evt_dict)` for each event.

    Async so we can `await asyncio.sleep(0)` after each emission — that
    yields control to the JS event loop so the browser can repaint
    per-node status and preview images while the (single-threaded)
    Pyodide runtime is between nodes. The JS side awaits the returned
    coroutine.
    """
    import asyncio as _asyncio
    _ABORT["flag"] = False
    req = json.loads(req_json)
    node_type_by_id = {n["id"]: n.get("type", "") for n in req["nodes"]}
    port_kinds_by_id = {
        nid: ({p.name: p.kind for p in CATALOG[t].outputs} if t in CATALOG else {})
        for nid, t in node_type_by_id.items()
    }
    try:
        for evt in engine.run_stream({"nodes": req["nodes"], "edges": req["edges"]}):
            if _ABORT["flag"]:
                on_event(json.dumps({"type": "error", "message": "aborted"}))
                return
            tag = evt[0]
            if tag == "node_start":
                on_event(json.dumps({"type": "node_start", "id": evt[1]}))
            elif tag == "node_progress":
                nid, prog, log = evt[1], evt[2], evt[3]
                payload: dict = {"type": "node_progress", "id": nid}
                if prog is not None:
                    payload["progress"] = float(prog)
                if log:
                    payload["log"] = str(log)
                on_event(json.dumps(payload))
            elif tag == "node_done":
                nid, res = evt[1], evt[2]
                kinds = port_kinds_by_id.get(nid, {})
                enc = _encode_node_result(nid, res, kinds)
                on_event(json.dumps({"type": "node_done", "id": nid, **enc}))
            elif tag == "done":
                on_event(json.dumps({"type": "done"}))
            # Hand the browser a chance to paint between every event.
            await _asyncio.sleep(0)
    except ValueError as e:
        on_event(json.dumps({"type": "error", "message": str(e)}))
    except Exception as e:  # noqa: BLE001
        on_event(json.dumps({"type": "error", "message": repr(e)}))
