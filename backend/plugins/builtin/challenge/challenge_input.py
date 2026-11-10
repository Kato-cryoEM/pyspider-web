"""Built-in node — data.challenge_input (Challenge · Fixture) — category: Data.

A read-only data source that resolves to one of the fixture images cached by
an active challenge session. The frontend places two of these on the canvas
when a challenge starts, with `session_id` filled in and `name` set to one
of the fixture names (e.g. "reference" or "shifted"). If the session is
missing (e.g. after a server restart) the node returns a visibly obvious
black image so the student can tell they need to restart the challenge.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, d
from backend.challenge_sessions import get_fixture


def _fn_challenge_input(session_id: str = "", name: str = ""):
    img = get_fixture(str(session_id), str(name))
    if img is None:
        # No session yet — return a 1×1 zero so downstream nodes error cleanly
        # rather than crashing on `None`. The frontend's challenge start wires
        # a real session_id before any graph run.
        img = np.zeros((1, 1), dtype=np.float32)
    return {"image": d(np.asarray(img, dtype=np.float32))}


register(N(
    "data.challenge_input", "Challenge · Fixture", "Data",
    "", "image:image",
    [
        ("session_id", "string", ""),
        ("name", "string", ""),
    ],
    "Read-only input image supplied by the current challenge session. "
    "The challenge start handshake fills `session_id`; `name` picks which "
    "fixture (e.g. reference / shifted). Normally the frontend places "
    "these nodes for you — you shouldn't drop one from the palette.",
    _fn_challenge_input,
))
