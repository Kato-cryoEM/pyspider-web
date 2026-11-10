"""Built-in node — stack.rotate (Rotate each frame) — category: Stack."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_rotate)

register(N(
    'stack.rotate', 'Rotate each frame', 'Stack',
    'stack:stack', 'stack:stack',
    [
        Param('angle_deg', 'float', 0.0, None, -360.0, 360.0, step=1.0),
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Rotate every frame of the stack by the same `angle_deg`. See RT for '
    'the single-image variant. `interpolation`: `nearest` / `linear` / '
    '`cubic` (default) / `quintic`.',
    _fn_stack_rotate))
