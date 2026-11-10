"""Stable public API for pyspider-web plugins.

Plugins should import ONLY from this module. Everything re-exported here is
part of a maintained contract; internal helpers (backend.nodes internals,
backend.formats._read_mrc, etc.) may move without notice.

Two extension surfaces:

  1. Nodes  — new operations that appear in the palette.

     ```python
     from backend.plugin_api import register, N, h, d
     import numpy as np

     def _fn_edge(image, sigma: float = 1.0):
         from scipy import ndimage as ndi
         return {"image": d(ndi.gaussian_filter(h(image), sigma))}

     register(
         N("my.edge", "MY · Sobel", "Custom",
           "image:image", "image:image",
           [("sigma", "float", 1.0, None, 0.1, 20.0)],
           "Gaussian smoothing demo plugin.",
           _fn_edge),
     )
     ```

  2. File formats — new readers / writers hooked into `io.load`, `io.save`,
     and the auto-save layer.

     ```python
     from backend.plugin_api import register_reader, register_writer
     import numpy as np, tifffile

     register_reader(".tif", lambda p: tifffile.imread(str(p)).astype(np.float32))
     register_writer(".tif", lambda p, a: tifffile.imwrite(str(p), a))
     ```
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# --- Node authoring helpers (re-exported) ---
from .nodes import NodeSpec, Port, Param, N, h, d, _fold_manual

# --- Format registry (re-exported) ---
from .formats import (
    register_reader,
    register_writer,
    read_array,
    write_array,
    readers,
    writers,
)

# --- Runtime backend (re-exported) ---
from . import xp

# Nodes registered by plugins live here; nodes.py merges them into CATALOG
# once plugins have finished loading.
_REGISTERED_NODES: list[NodeSpec] = []


def register(spec: NodeSpec) -> None:
    """Add a node to the plugin catalog. Called at plugin import time."""
    if not isinstance(spec, NodeSpec):
        raise TypeError(f"register() expects a NodeSpec, got {type(spec).__name__}")
    _REGISTERED_NODES.append(spec)


def registered_nodes() -> list[NodeSpec]:
    """Snapshot of nodes contributed by plugins so far."""
    return list(_REGISTERED_NODES)


def enrich_manual(spec: NodeSpec) -> None:
    """Optional: fold the local SPIDER manual excerpt into a node's doc
    string. Only relevant when the plugin sets `manual=<SPIDER cmd>`."""
    _fold_manual(spec)


__all__ = [
    "NodeSpec", "Port", "Param", "N", "h", "d",
    "register", "registered_nodes",
    "register_reader", "register_writer",
    "read_array", "write_array", "readers", "writers",
    "xp", "np", "enrich_manual",
]
