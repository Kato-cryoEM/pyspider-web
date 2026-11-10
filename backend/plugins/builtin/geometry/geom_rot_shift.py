"""Built-in node — geom.rot_shift (Rotate + Shift) — category: Geometry."""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_rot_shift)

register(N(
    'geom.rot_shift', 'Rotate + Shift', 'Geometry',
    'image:image', 'image:image',
    [
        Param('angle_deg', 'float', 0.0, None, -360.0, 360.0, step=1.0),
        ('dx', 'float', 0.0),
        ('dy', 'float', 0.0),
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Combine rotation + translation into ONE interpolation '
    'pass (minimises blurring vs. rotating then shifting separately). '
    'Parameters as in RT + SH. `interpolation` picks the spline kind for '
    'both operations.',
    _fn_rot_shift, manual='RT SQ'))
