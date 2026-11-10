"""Persist node outputs to disk.

Format policy:
  - image / volume / stack (2D or 3D numpy arrays)  →  MRC (.mrc, default)
    via the shared format registry — plugins can override MRC or add other
    array formats (.tif, .h5, ...) transparently.
  - doc (list of numeric rows)                      →  STAR loop_ (.star)
  - scalar / string                                 →  STAR (.star)
  - anything else                                   →  numpy binary (.npy)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from . import formats, xp


# ---- STAR writer -----------------------------------------------------------

def _write_star_loop(path: Path, rows: list[list[float]],
                     col_names: list[str] | None = None,
                     header_comment: str = "") -> None:
    if not rows:
        n_cols = 0
    else:
        n_cols = max(len(r) for r in rows)
    if not col_names:
        col_names = [f"rlnCol{i + 1}" for i in range(n_cols)]

    with path.open("w") as f:
        f.write("# pyspider-web STAR output\n")
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        f.write("\ndata_\n\nloop_\n")
        for i, name in enumerate(col_names, 1):
            f.write(f"_{name} #{i}\n")
        for row in rows:
            padded = list(row) + [0.0] * (n_cols - len(row))
            f.write(" ".join(f"{v:14.6g}" for v in padded) + "\n")


def _write_star_scalar(path: Path, name: str, value: Any) -> None:
    with path.open("w") as f:
        f.write("# pyspider-web STAR output\n\n")
        f.write("data_\n\n")
        if isinstance(value, (int, float, bool)):
            f.write(f"_rln{name.capitalize()} {value!s}\n")
        else:
            # Escape as a semicolon-delimited multi-line block
            s = str(value)
            if "\n" in s:
                f.write(f"_rln{name.capitalize()}\n")
                f.write(";\n" + s.rstrip() + "\n;\n")
            else:
                # single-line strings: quote if they contain spaces
                if any(c.isspace() for c in s):
                    f.write(f"_rln{name.capitalize()} \"{s}\"\n")
                else:
                    f.write(f"_rln{name.capitalize()} {s}\n")


def _try_parse_numeric_table(text: str) -> list[list[float]] | None:
    lines = [l for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    rows: list[list[float]] = []
    for l in lines:
        parts = [p for p in re.split(r"[,\s]+", l.strip()) if p]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            return None
    return rows or None


# ---- Public entry point ----------------------------------------------------

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize(s: str) -> str:
    return _SAFE_RE.sub("_", str(s))


def save_output(node_id: str, port_name: str,
                port_kind: str | None,
                value: Any, dir_path: str) -> str:
    """Persist ``value`` to ``dir_path`` and return the file path written.

    Format is chosen by value type / port_kind:
      * numpy arrays → MRC (2D or 3D). Complex arrays fall back to .npy.
      * lists (doc)  → STAR loop_
      * scalars/str  → STAR (data_ block)
      * anything else → .npy
    """
    d = Path(dir_path).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = f"{_sanitize(node_id)}_{_sanitize(port_name)}"

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = xp.to_host(value)
        if np.iscomplexobj(arr):
            path = d / (base + ".npy")
            np.save(path, arr)
            return str(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim in (2, 3):
            # Dispatch through the format registry: default is MRC, but a
            # plugin can register another writer for .mrc / add new exts.
            path = d / (base + ".mrc")
            formats.write_array(path, arr)
            return str(path)
        path = d / (base + ".npy")
        np.save(path, arr)
        return str(path)

    if isinstance(value, list):
        rows: list[list[float]] = []
        for r in value:
            if isinstance(r, (list, tuple)):
                try:
                    rows.append([float(x) for x in r])
                except (TypeError, ValueError):
                    pass
        path = d / (base + ".star")
        _write_star_loop(path, rows,
                         header_comment=f"node={node_id} port={port_name}")
        return str(path)

    if isinstance(value, str):
        parsed = _try_parse_numeric_table(value)
        path = d / (base + ".star")
        if parsed is not None:
            _write_star_loop(path, parsed,
                             header_comment=f"node={node_id} port={port_name}")
        else:
            _write_star_scalar(path, port_name, value)
        return str(path)

    if isinstance(value, (int, float, bool)):
        path = d / (base + ".star")
        _write_star_scalar(path, port_name, value)
        return str(path)

    path = d / (base + ".npy")
    np.save(path, np.asarray(value, dtype=object))
    return str(path)
