"""Built-in node — data.pt (PT · Pattern image) — category: Data.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_pt)

# Which `pattern` codes use which coordinate slots. Matches SPIDER PT.
_USE_X1     = ['point', 'line', 'cross',
               'filled_circle', 'outlined_circle', 'circle_3pts',
               'filled_triangle', 'outlined_triangle',
               'filled_box', 'outlined_box', 'rect_outline']
_USE_X2     = ['line', 'circle_3pts',
               'filled_triangle', 'outlined_triangle',
               'filled_box', 'outlined_box', 'rect_outline']
_USE_X3     = ['circle_3pts', 'filled_triangle', 'outlined_triangle']
_USE_RADIUS = ['filled_circle', 'outlined_circle']

register(N(
    'data.pt', 'Pattern image', 'Data', '', 'image:image',
    [
        Param('nx', 'int', 128, None, 4, 8192),
        Param('ny', 'int', 128, None, 4, 8192),
        Param('pattern', 'select', 'point', [
            'point', 'line',
            'filled_circle', 'outlined_circle', 'circle_3pts',
            'filled_triangle', 'outlined_triangle',
            'filled_box', 'outlined_box',
            'cross', 'rect_outline',
        ]),
        Param('x1', 'float', 32.0, depends_on=('pattern', _USE_X1)),
        Param('y1', 'float', 32.0, depends_on=('pattern', _USE_X1)),
        Param('x2', 'float', 96.0, depends_on=('pattern', _USE_X2)),
        Param('y2', 'float', 96.0, depends_on=('pattern', _USE_X2)),
        Param('x3', 'float', 32.0, depends_on=('pattern', _USE_X3)),
        Param('y3', 'float', 96.0, depends_on=('pattern', _USE_X3)),
        Param('radius', 'float', 16.0, None, 0.0, 8192.0,
              depends_on=('pattern', _USE_RADIUS)),
        Param('intensity', 'float', 1.0),
        Param('background', 'float', 0.0),
        Param('subtract', 'bool', False),
    ],
    'Build a 2D pattern / mask image by drawing an elementary '
    'shape onto a `background` field. Pattern codes match SPIDER: '
    'point (P), line (L), filled_circle (C), outlined_circle (CL), '
    'circle_3pts (CJ = unique circle through 3 points), '
    'filled_triangle (T), outlined_triangle (TL), '
    'filled_box (B), outlined_box (BL). '
    'subtract = SPIDER "S" suffix: writes `background` instead of '
    '`intensity`, i.e. erases the shape from an existing mask. '
    'Extras beyond SPIDER: cross (full-image cross-hair), rect_outline '
    '(alias of outlined_box). Coordinate/radius fields show/hide in real '
    'time based on the selected pattern.',
    _fn_pt, manual='PT'))
