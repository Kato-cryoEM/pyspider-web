"""FastAPI entrypoint for the pyspider node editor.

Endpoints:
  GET  /api/health          runtime status (backend + cuda availability)
  GET  /api/nodes           node catalog for the palette
  POST /api/backend         switch cpu/cuda at runtime
  POST /api/graph/run       execute a graph, return per-node previews

Static UI:
  GET  /                    serves the pre-built React app (frontend/dist),
                            if that directory exists. Otherwise only the API
                            is available and users need to run `npm run dev`.

Run with:
  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import challenge_sessions, engine, formats, plugin_loader, preview as preview_mod, save as save_mod, settings, xp
from .nodes import CATALOG, public_catalog
from .preview import encode_output


# backend/main.py → backend/ → project_root/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The pre-built UI ships inside the backend package as `_static/` (that's
# where vite.config.ts writes to and what pip bundles). When developing from
# a source checkout without a build yet, we also accept the legacy
# `frontend/dist/` alongside the project — first hit wins.
_STATIC_CANDIDATES = [
    Path(__file__).resolve().parent / "_static",   # installed / built
    _PROJECT_ROOT / "frontend" / "dist",           # legacy dev location
]
_FRONTEND_DIST = next((p for p in _STATIC_CANDIDATES if p.is_dir()),
                     _STATIC_CANDIDATES[0])


app = FastAPI(title="pyspider node editor", version="0.1.0")

# Vite dev server runs on 5173; allow it plus 127.0.0.1 variants.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BackendReq(BaseModel):
    name: str  # "cpu" | "cuda"


class GraphReq(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@app.get("/api/health")
def health() -> dict:
    return {
        "backend": xp.current(),
        "cuda_available": xp.has_cuda(),
    }


@app.get("/api/nodes")
def nodes() -> dict:
    return {"nodes": public_catalog()}


@app.get("/api/plugins")
def plugins() -> dict:
    """Report plugin discovery status and current file-format coverage."""
    return {
        "plugins": plugin_loader.load_log(),
        "readers": formats.readers(),
        "writers": formats.writers(),
    }


# --- Tutorials ------------------------------------------------------------

# Lessons live as JSON files so anyone (not just TypeScript authors) can add
# one. Two discovery locations, mirroring the plugin loader convention:
#   1. `backend/tutorials/` (bundled with the app — reference / official)
#   2. `~/.pyspider/tutorials/`  (user-scoped — survives upgrades)
_BUILTIN_TUTORIAL_DIR = _PROJECT_ROOT / "backend" / "tutorials"
_USER_TUTORIAL_DIR = Path.home() / ".pyspider" / "tutorials"


def _load_tutorials() -> tuple[list[dict], list[dict]]:
    """Return (lessons, load_errors). Errors are reported inline so a broken
    lesson doesn't hide the working ones."""
    lessons: list[dict] = []
    errors: list[dict] = []
    seen_ids: set[str] = set()
    for d in (_BUILTIN_TUTORIAL_DIR, _USER_TUTORIAL_DIR):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                errors.append({"file": str(f), "error": str(e)})
                continue
            # Minimum-schema sanity check so a typo in a user lesson doesn't
            # break the whole tutorial menu.
            if not (isinstance(obj, dict)
                    and isinstance(obj.get("id"), str)
                    and isinstance(obj.get("title"), str)
                    and isinstance(obj.get("steps"), list)):
                errors.append({"file": str(f),
                               "error": "missing id / title / steps"})
                continue
            if obj["id"] in seen_ids:
                # User dir overrides built-in with same id: keep whichever we
                # loaded first (built-in dir is scanned first, but user files
                # can shadow with a different filename + same id — flag it).
                errors.append({"file": str(f),
                               "error": f"duplicate lesson id {obj['id']!r}"})
                continue
            seen_ids.add(obj["id"])
            lessons.append(obj)
    return lessons, errors


class PredictShapesReq(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@app.post("/api/predict_shapes")
def predict_shapes(req: PredictShapesReq) -> dict:
    """Predict each node's primary image-output shape given the current
    graph — without running anything. The frontend uses these shapes to
    size the pre-Run preview slot so the node's box is already the right
    aspect ratio (clock 256² → PO 128×~804) before execution."""
    from .predict import predict_shapes as _predict
    return {"shapes": _predict({"nodes": req.nodes, "edges": req.edges})}


@app.get("/api/tutorials")
def tutorials() -> dict:
    """Return the current list of tutorial lessons. Read from disk on every
    request so an author editing JSON sees changes without a server restart."""
    lessons, errors = _load_tutorials()
    return {"lessons": lessons, "errors": errors}


class TutorialSave(BaseModel):
    lesson: dict


@app.post("/api/tutorials/save")
def save_tutorial(req: TutorialSave) -> dict:
    """Persist a lesson JSON to the user tutorial dir. The Tutorial Maker
    posts here; the filename is derived from the lesson id so re-saves
    overwrite the same file. Built-in lessons are never overwritten — we
    always write to the user dir so upgrades don't clobber user edits."""
    l = req.lesson
    if not (isinstance(l, dict)
            and isinstance(l.get("id"), str) and l["id"]
            and isinstance(l.get("title"), str)
            and isinstance(l.get("steps"), list)):
        raise HTTPException(status_code=400,
            detail="lesson must have id, title, and steps[]")
    # Sanitise the id → filename (no slashes, no dots).
    safe = "".join(c for c in l["id"] if c.isalnum() or c in "-_") or "tutorial"
    _USER_TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _USER_TUTORIAL_DIR / f"{safe}.json"
    path.write_text(json.dumps(l, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "id": l["id"]}


# --- Challenges -----------------------------------------------------------
# A challenge is a short puzzle: fixed input images, a limited palette, and
# a goal to reach. The backend caches the ground-truth images under a session
# id so student graphs can resolve `data.challenge_input` nodes by key.


@app.get("/api/challenges")
def list_challenges() -> dict:
    """Return the public summary of every challenge spec on disk."""
    specs, errors = challenge_sessions.load_challenges()
    # Strip judging internals from the listing — those stay server-side.
    public = [
        {k: v for k, v in s.items()
         if k not in ("compare_to", "metric", "pass_score")}
        for s in specs
    ]
    return {"challenges": public, "errors": errors}


class ChallengeStart(BaseModel):
    challenge_id: str


@app.post("/api/challenges/start")
def start_challenge(req: ChallengeStart) -> dict:
    spec = challenge_sessions.get_spec(req.challenge_id)
    if spec is None:
        raise HTTPException(status_code=404,
            detail=f"challenge {req.challenge_id!r} not found")
    try:
        return challenge_sessions.start_session(spec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ChallengeFixturePreview(BaseModel):
    session_id: str
    name: str


@app.post("/api/challenges/fixture_preview")
def fixture_preview(req: ChallengeFixturePreview) -> dict:
    """Return a PNG-encoded preview of a session fixture so the prompt panel
    can show the student what they're working with before they wire anything."""
    img = challenge_sessions.get_fixture(req.session_id, req.name)
    if img is None:
        raise HTTPException(status_code=404,
            detail=f"no fixture {req.name!r} in session {req.session_id!r}")
    # 3D fixtures preview as volumes (middle-slice thumbnail is still filled
    # into `png`); 2D ones stay plain images. 1D fixtures (e.g. the hidden
    # per-frame class labels used by the clustering judge) have no useful
    # thumbnail — report shape only so the panel shows a placeholder.
    nd = getattr(img, "ndim", 0)
    if nd < 2:
        return {"png": {"kind": "scalar", "shape": list(getattr(img, "shape", ())),
                        "dtype": str(getattr(img, "dtype", "")),
                        "png": None, "min": None, "max": None}}
    kind = "volume" if nd == 3 else "image"
    return {"png": encode_output(img, kind=kind, port_name=req.name)}


class ChallengeJudgeReq(BaseModel):
    challenge_id: str
    session_id: str
    answer_node_id: str
    nodes: list[dict]
    edges: list[dict]


@app.post("/api/challenges/judge")
def judge_challenge(req: ChallengeJudgeReq) -> dict:
    """Run the student's graph and compare the designated answer against the
    reference fixture using the metric declared in the challenge JSON."""
    spec = challenge_sessions.get_spec(req.challenge_id)
    if spec is None:
        raise HTTPException(status_code=404,
            detail=f"challenge {req.challenge_id!r} not found")
    # TEMP debug dump of submitted graph
    try:
        import json as _json
        _dump = {"challenge_id": req.challenge_id,
                 "answer_node_id": req.answer_node_id,
                 "nodes": [{"id": n.get("id"), "type": n.get("type"),
                            "parent": n.get("parent"),
                            "params": n.get("params", {})}
                           for n in req.nodes],
                 "edges": req.edges}
        with open("/tmp/last_judge.json", "w") as _f:
            _json.dump(_dump, _f, indent=2, default=str)
    except Exception:
        pass
    # Defense in depth: reject graphs that read from a judge-only fixture.
    # The frontend already hides those from the canvas, but a student could
    # still hand-craft a data.challenge_input node that names the locked
    # fixture. This blocks the "wire the answer directly" short-circuit.
    _judge_only_names = {
        f["name"] for f in spec.get("fixtures", [])
        if f.get("judge_only")
    }
    if _judge_only_names:
        _locked_sources: set[str] = set()
        for n in req.nodes:
            if n.get("type") not in ("data.challenge_input",
                                     "data.challenge_input_volume"):
                continue
            name = (n.get("params") or {}).get("name")
            if name in _judge_only_names:
                _locked_sources.add(n.get("id"))
        if _locked_sources:
            for e in req.edges:
                if e.get("src") in _locked_sources:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"challenge rule: fixture is locked "
                            f"— cannot be used as a graph input (judge only)"
                        ))
    try:
        results = engine.run({"nodes": req.nodes, "edges": req.edges})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"graph error: {e}")
    res = results.get(req.answer_node_id)
    if res is None or not res.ok:
        err = res.error if res else "answer node did not run"
        return {"ok": False, "pass": False, "score": None,
                "error": f"answer node failed: {err}"}
    import numpy as _np
    port = spec.get("answer_port", "image")
    # Flexible port resolution. Priority: (1) exact name match, (2) known
    # name aliases, (3) fall-through to any port of the right kind. This
    # lets students submit terminal nodes with idiosyncratic output names
    # (flow.state.value_out, ap_shc.reference, …) without memorising the
    # challenge-specific port alias.
    aliases = {"image":  ["image", "value_out", "stack", "volume", "cc",
                           "reference", "aligned", "result", "out"],
               "stack":  ["stack", "class_means", "aligned", "value_out",
                          "image"],
               "volume": ["volume", "stack"],
               # Label-stream ports: the per-frame integer class assignment
               # produced by ap_mr / ap_md and consumed by the clustering
               # judge. The exact name varies, so we accept any of them.
               "labels": ["ref_index", "labels", "class_ids",
                          "assignments"]}
    picked_port = None
    for alt in aliases.get(port, [port]):
        if alt in res.outputs:
            picked_port = alt
            break
    # Last resort: pick any array-shaped output. For `image` challenges we
    # accept 2D/3D (3D auto-flattens below); for `stack` challenges we
    # require 3D; for `labels` challenges we accept 1D int arrays.
    if picked_port is None:
        for name, v in res.outputs.items():
            if not hasattr(v, "shape"):
                continue
            nd = getattr(v, "ndim", 0)
            if port == "image" and nd in (2, 3):
                picked_port = name
                break
            if port == "stack" and nd == 3:
                picked_port = name
                break
            if port == "labels" and nd == 1:
                picked_port = name
                break
    if picked_port is None:
        return {"ok": False, "pass": False, "score": None,
                "error": f"answer node has no `{port}` output "
                         f"(has: {sorted(res.outputs.keys())})"}
    val = _np.asarray(res.outputs[picked_port])
    # Auto-flatten 3D → 2D when the challenge expects a 2D image.
    if port == "image" and val.ndim == 3:
        val = val[val.shape[0] // 2]
    # For label-stream challenges, cast to int so the judge gets clean
    # label values (ap_mr returns int32 already, but be defensive).
    if port == "labels":
        val = val.astype(int)
    return challenge_sessions.judge(spec, req.session_id, val)


@app.post("/api/backend")
def switch_backend(req: BackendReq) -> dict:
    try:
        xp.set_backend(req.name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"backend": xp.current()}


class SettingsPatch(BaseModel):
    default_save_dir: str | None = None


@app.get("/api/settings")
def get_settings() -> dict:
    return settings.get_settings()


@app.get("/api/browse")
def browse(path: str = "") -> dict:
    """List a directory on the server's filesystem (for the save-dir picker).

    Hidden entries (dotfiles) are hidden by default. Non-existent paths fall
    back to the user's home directory so the picker always opens on something
    real even when the current text input is broken.
    """
    p = Path(path).expanduser() if path else Path.home()
    if not p.exists():
        p = Path.home()
    p = p.resolve()
    if not p.is_dir():
        p = p.parent
    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except (PermissionError, OSError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    dirs = [e.name for e in entries
            if e.is_dir() and not e.name.startswith(".")]
    files = [e.name for e in entries
             if e.is_file() and not e.name.startswith(".")]
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}


@app.post("/api/settings")
def post_settings(patch: SettingsPatch) -> dict:
    updates = {k: v for k, v in patch.model_dump().items() if v is not None}
    try:
        return settings.update_settings(updates)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _apply_saves(node_id: str, node_saves: dict | None,
                 res, port_kinds: dict[str, str],
                 default_dir: str) -> dict[str, str]:
    """For each port with save enabled, persist the value and return
    {port_name: path_or_error_string}."""
    saved: dict[str, str] = {}
    if not (res and res.ok and node_saves):
        return saved
    for port_name, cfg in node_saves.items():
        if not isinstance(cfg, dict):
            continue
        if port_name not in res.outputs:
            continue
        dir_path = (cfg.get("dir") or "").strip() or default_dir
        try:
            path = save_mod.save_output(
                node_id, port_name, port_kinds.get(port_name),
                res.outputs[port_name], dir_path,
            )
            saved[port_name] = path
        except Exception as e:  # noqa: BLE001
            saved[port_name] = f"ERROR: {e}"
    return saved


@app.post("/api/graph/run")
def run_graph(req: GraphReq) -> dict:
    try:
        results = engine.run({"nodes": req.nodes, "edges": req.edges})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    node_type_by_id = {n["id"]: n.get("type", "") for n in req.nodes}
    node_params_by_id = {n["id"]: (n.get("params") or {}) for n in req.nodes}
    save_by_id = {n["id"]: n.get("save") for n in req.nodes}
    default_dir = settings.get_settings().get("default_save_dir", "")

    payload: dict[str, dict] = {}
    for nid, res in results.items():
        ntype = node_type_by_id.get(nid, "")
        nparams = node_params_by_id.get(nid, {})
        spec = CATALOG.get(ntype)
        port_kinds = {p.name: p.kind for p in spec.outputs} if spec else {}
        entry = _encode_node_result(nid, res, port_kinds, ntype, nparams)
        saved = _apply_saves(nid, save_by_id.get(nid), res, port_kinds, default_dir)
        if saved:
            entry["saved"] = saved
        payload[nid] = entry
    return {"results": payload}


def _encode_node_result(nid: str, res, port_kinds: dict[str, str],
                        node_type: str = "",
                        node_params: dict | None = None) -> dict:
    if res is None:
        return {"ok": False, "error": "no result", "outputs": {}}
    out: dict[str, dict] = {}
    for port_name, value in res.outputs.items():
        out[port_name] = encode_output(value, kind=port_kinds.get(port_name),
                                        port_name=port_name)

    # Note: the frontend renders the 2D filter response as a live canvas
    # (FilterRing.tsx) driven by the current params, so no backend PNG is
    # attached here — that avoids a round-trip on every param tweak.
    return {"ok": res.ok, "error": res.error, "outputs": out}


@app.post("/api/graph/run_stream")
def run_graph_stream(req: GraphReq):
    """Same as /api/graph/run, but streams NDJSON events per node.

    Event schema (one JSON object per line):
      {"type":"node_start", "id":"n1"}
      {"type":"node_done",  "id":"n1", "ok":true, "error":null, "outputs":{...}}
      {"type":"done"}                    end of stream
      {"type":"error",   "message":"..."} fatal engine error (rare)

    Inside a DO container, children emit start/done on every iteration so the
    UI can flash their status per pass.
    """
    node_type_by_id = {n["id"]: n.get("type", "") for n in req.nodes}
    node_params_by_id = {n["id"]: (n.get("params") or {}) for n in req.nodes}
    save_by_id = {n["id"]: n.get("save") for n in req.nodes}
    port_kinds_by_id = {
        nid: ({p.name: p.kind for p in CATALOG[t].outputs} if t in CATALOG else {})
        for nid, t in node_type_by_id.items()
    }
    default_dir = settings.get_settings().get("default_save_dir", "")

    def _gen():
        try:
            for evt in engine.run_stream({"nodes": req.nodes, "edges": req.edges}):
                tag = evt[0]
                if tag == "node_start":
                    yield json.dumps({"type": "node_start", "id": evt[1]}) + "\n"
                elif tag == "node_progress":
                    nid, prog, log = evt[1], evt[2], evt[3]
                    payload = {"type": "node_progress", "id": nid}
                    if prog is not None:
                        payload["progress"] = float(prog)
                    if log:
                        payload["log"] = str(log)
                    yield json.dumps(payload) + "\n"
                elif tag == "node_done":
                    nid, res = evt[1], evt[2]
                    kinds = port_kinds_by_id.get(nid, {})
                    enc = _encode_node_result(
                        nid, res, kinds,
                        node_type=node_type_by_id.get(nid, ""),
                        node_params=node_params_by_id.get(nid, {}),
                    )
                    saved = _apply_saves(nid, save_by_id.get(nid), res, kinds, default_dir)
                    if saved:
                        enc["saved"] = saved
                    yield json.dumps({"type": "node_done", "id": nid, **enc}) + "\n"
                elif tag == "done":
                    yield json.dumps({"type": "done"}) + "\n"
        except ValueError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "error", "message": repr(e)}) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


# Mount the pre-built React app LAST so /api/* routes take precedence.
# When frontend/dist is absent (fresh checkout with no build), skip silently
# and let the developer run `vite dev` on 5173 in the traditional flow.
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True),
              name="frontend")
