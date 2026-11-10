"""Built-in node — geom.shift (Shift) — category: Geometry."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_shift)

register(N(
    'geom.shift', 'Shift', 'Geometry',
    'image:image', 'image:image',
    [
        ('dx', 'float', 0.0),
        ('dy', 'float', 0.0),
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Translate the image by sub-pixel (dx, dy). Positive `dx` '
    'moves content right; positive `dy` moves it down. `cval` fills '
    'freshly-exposed border pixels. `interpolation`: `nearest` / `linear` / '
    '`cubic` (default) / `quintic` — trades speed for smoothness.',
    _fn_shift, manual='SH'))
