"""Backend switch: numpy (CPU) vs cupy (CUDA).

Pattern: modules that want to be GPU-capable import ``xp`` and use
``xp.get_array_module(a)`` to dispatch — the same idiom cupy documents.
When cupy is not installed, ``get_array_module`` always returns numpy
and ``to_device`` is a no-op.  This lets the same pyspider function
run on CPU or GPU without code duplication.

The active *default* device is set by ``set_backend("cpu" | "cuda")``
and read by ``current()``.  It only affects fresh array creation
inside nodes — arrays that already live on a device stay there.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as _cupy  # type: ignore
    _HAS_CUPY = True
except Exception:
    _cupy = None
    _HAS_CUPY = False


_BACKEND = "cuda" if _HAS_CUPY else "cpu"


def has_cuda() -> bool:
    return _HAS_CUPY


def set_backend(name: str) -> str:
    global _BACKEND
    if name == "cuda" and not _HAS_CUPY:
        raise RuntimeError("cupy is not installed; cannot select 'cuda' backend")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unknown backend: {name!r}")
    _BACKEND = name
    return _BACKEND


def current() -> str:
    return _BACKEND


def module():
    """Return the active array module (numpy or cupy)."""
    return _cupy if _BACKEND == "cuda" else np


def get_array_module(a):
    """Dispatch on an existing array's device (cupy idiom)."""
    if _HAS_CUPY and _cupy.get_array_module(a) is _cupy:
        return _cupy
    return np


def to_device(a):
    """Move array to the active backend, no copy if already there."""
    xp = module()
    if xp is np:
        return np.asarray(a) if not isinstance(a, np.ndarray) else a
    if isinstance(a, _cupy.ndarray):
        return a
    return _cupy.asarray(a)


def to_host(a) -> np.ndarray:
    """Bring array back to CPU as numpy."""
    if _HAS_CUPY and isinstance(a, _cupy.ndarray):
        return _cupy.asnumpy(a)
    return np.asarray(a)
