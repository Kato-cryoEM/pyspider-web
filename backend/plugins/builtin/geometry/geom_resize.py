"""Built-in node — geom.resize (Interpolate/Resize) — category: Geometry."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_resize)

register(N(
    'geom.resize', 'Interpolate/Resize', 'Geometry',
    'image:image', 'image:image',
    [
        ('ny', 'int', 128, None, 8, 8192),
        ('nx', 'int', 128, None, 8, 8192),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
        ('anti_alias', 'select', 'auto', ['auto', 'yes', 'no']),
    ],
    'Resample the image to size (ny, nx). '
    '`interpolation` picks the spline kind: `nearest` (fastest, blocky), '
    '`linear` (bilinear), `cubic` (default, smooth), `quintic` (highest). '
    '`anti_alias` applies a low-pass filter before down-sampling to prevent '
    'aliasing; `auto` enables it only when shrinking.',
    _fn_resize, manual='IP'))
