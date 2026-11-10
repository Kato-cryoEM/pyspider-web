"""Built-in node — geom.rotate (Rotate) — category: Geometry."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_rotate)

register(N(
    'geom.rotate', 'Rotate', 'Geometry',
    'image:image', 'image:image',
    [
        Param('angle_deg', 'float', 0.0, None, -360.0, 360.0, step=1.0),
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Rotate about the image centre by `angle_deg` (positive = '
    'counter-clockwise). `cval` fills pixels that fall outside the source '
    'image after rotation. `interpolation` picks the spline kind: '
    '`nearest` (fastest, blocky), `linear` (bilinear), `cubic` (default — '
    'smooth, good default for cryo-EM), `quintic` (slowest, marginally '
    'sharper).',
    _fn_rotate, manual='RT'))
