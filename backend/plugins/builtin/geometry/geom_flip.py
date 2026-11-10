"""Built-in node — geom.flip (Mirror along one axis) — Geometry.

Consolidates `geom.flip_h` and `geom.flip_v` into a single node with an
`axis` dropdown. Both former nodes had identical I/O (image → image) and
no parameters; only the axis of mirroring differed. One node with a
dropdown lets the user flip the mirroring direction without replacing
the node and rewiring.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _fn_flip_h, _fn_flip_v


_AXES = ["horizontal", "vertical"]

_FN = {
    "horizontal": _fn_flip_h,
    "vertical":   _fn_flip_v,
}


def _fn_flip(image, axis: str = "horizontal"):
    fn = _FN.get(axis)
    if fn is None:
        raise ValueError(f"unknown axis: {axis!r}. Known: {_AXES}")
    return fn(image)


_DOC = (
    "Mirror the image along one axis. The `axis` dropdown picks the "
    "direction of mirroring:\n\n"
    "  • `horizontal` — flip left-right (mirror across the vertical "
    "centre line). Equivalent to SPIDER MR with axis = x.\n"
    "  • `vertical`   — flip top-bottom (mirror across the horizontal "
    "centre line). Equivalent to SPIDER MR with axis = y.\n\n"
    "Applied before or after rotation, this is a cheap way to generate "
    "mirror-pair references for alignment / classification routines "
    "that don't test the mirror internally."
)


register(N(
    "geom.flip", "Flip (mirror)", "Geometry",
    "image:image", "image:image",
    [Param("axis", "select", "horizontal", _AXES)],
    _DOC, _fn_flip, manual="MR"))
