"""Built-in node — data.challenge_input_volume — Data.

Volume-typed sibling of ``data.challenge_input``. The frontend places one
of these for every challenge fixture whose JSON declares
``"kind": "volume"``; the resolver and session cache are shared.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, d
from backend.challenge_sessions import get_fixture


def _fn_challenge_input_volume(session_id: str = "", name: str = ""):
    vol = get_fixture(str(session_id), str(name))
    if vol is None:
        # No active session — hand back a tiny zero volume so downstream
        # nodes error on shape rather than crashing on `None`.
        vol = np.zeros((1, 1, 1), dtype=np.float32)
    arr = np.asarray(vol, dtype=np.float32)
    if arr.ndim != 3:
        # Fixture was built as 2D (image) but wired through the volume
        # plugin — promote to a length-1 stack-shaped volume so the error
        # message downstream is at least shape-based.
        arr = arr[np.newaxis, ...]
    return {"volume": d(arr)}


register(N(
    "data.challenge_input_volume", "Challenge · Fixture (volume)", "Data",
    "", "volume:volume",
    [
        ("session_id", "string", ""),
        ("name", "string", ""),
    ],
    "Read-only 3D volume supplied by the current challenge session. "
    "Volume-typed counterpart of data.challenge_input — the frontend "
    "picks whichever one matches the fixture's `kind` field.",
    _fn_challenge_input_volume,
))
