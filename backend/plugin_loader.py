"""Drop-in plugin loader.

Discovers Python files in:
  1. ``backend/plugins/`` (bundled with the app)
  2. every path in the ``PYSPIDER_PLUGIN_DIR`` env var (os.pathsep-separated)
  3. ``~/.pyspider/plugins/`` (user-scope, survives upgrades)

Each ``*.py`` file is imported once. During import, plugin code calls
``plugin_api.register(...)`` and/or ``plugin_api.register_reader/writer(...)``
to contribute nodes and format handlers.

Files that begin with ``_`` are skipped (private helpers). Load failures are
captured in a log accessible via ``load_log()`` so the frontend can show
which plugins refused to start without derailing the rest of the catalog.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_BUILTIN_DIR = Path(__file__).resolve().parent / "plugins"

_LOAD_LOG: list[dict] = []


def _extra_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("PYSPIDER_PLUGIN_DIR", "").strip()
    if env:
        for p in env.split(os.pathsep):
            if p.strip():
                dirs.append(Path(p.strip()).expanduser())
    dirs.append(Path.home() / ".pyspider" / "plugins")
    return dirs


def load_plugins() -> list[dict]:
    """Import every non-private .py under the plugin dirs (recursively).

    Subdirectories are walked so authors can group related plugins together
    (e.g. backend/plugins/builtin/io/load.py). Files or directories that
    start with '_' are skipped (private helpers). Returns the load log.
    """
    _LOAD_LOG.clear()
    seen: set[Path] = set()
    for d in [_BUILTIN_DIR, *_extra_dirs()]:
        try:
            if not d.is_dir():
                continue
        except OSError:
            continue
        for py in sorted(d.rglob("*.py")):
            if py.name.startswith("_"):
                continue
            # Skip anything under a directory whose name starts with '_'
            if any(part.startswith("_") for part in py.relative_to(d).parts[:-1]):
                continue
            real = py.resolve()
            if real in seen:
                continue
            seen.add(real)
            _load_one(py)
    return list(_LOAD_LOG)


def _load_one(path: Path) -> None:
    mod_name = f"_pyspider_plugin_{path.stem}_{abs(hash(str(path))) & 0xFFFF}"
    entry: dict = {"path": str(path), "ok": False, "error": None, "nodes_before": 0, "nodes_added": 0}
    try:
        from . import plugin_api
        entry["nodes_before"] = len(plugin_api._REGISTERED_NODES)  # noqa: SLF001
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        entry["nodes_added"] = len(plugin_api._REGISTERED_NODES) - entry["nodes_before"]  # noqa: SLF001
        entry["ok"] = True
    except Exception as e:  # noqa: BLE001
        entry["error"] = repr(e)
    _LOAD_LOG.append(entry)


def load_log() -> list[dict]:
    return list(_LOAD_LOG)
