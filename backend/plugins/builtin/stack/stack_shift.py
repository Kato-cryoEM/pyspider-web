"""Built-in node — stack.shift (Shift each frame) — category: Stack."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_shift)

register(N(
    'stack.shift', 'Shift each frame', 'Stack',
    'stack:stack', 'stack:stack',
    [
        ('dx', 'float', 0.0),
        ('dy', 'float', 0.0),
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Shift every frame of the stack by the same (dx, dy). See SH for the '
    'single-image variant. `interpolation`: `nearest` / `linear` / `cubic` '
    '(default) / `quintic`.',
    _fn_stack_shift))
