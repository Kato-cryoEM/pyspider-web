"""Built-in node — mask.circle (Circular mask) — category: Masking.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_circle_mask)

register(N(
    'mask.circle', 'Circular mask', 'Masking',
    'image:image',
    'image:image',
    [
        Param('outer_radius', 'float', 30.0, None, 0.0, 8192.0, step=1.0),
        Param('inner_radius', 'float', 0.0, None, 0.0, 8192.0, step=1.0),
        Param('edge', 'select', 'disk',
              ['disk', 'cosine', 'gauss_edge', 'true_gauss']),
        Param('edge_width', 'float', 3.0, None, 0.1, 100.0, step=1.0),
        Param('background_mode', 'select', 'average',
              ['average', 'circumference', 'external']),
        Param('background', 'float', 0.0),
    ],
    'Circular mask applied to an image. All radii are in '
    'pixels of the input image. '
    '**outer_radius** (default 30) keeps pixels inside a disk of this '
    'radius; pixels outside are replaced by `background_mode`. Use 0 to '
    'auto-fit to half the shorter image side. '
    '**inner_radius** > 0 punches out an inner hole (annulus mask). '
    '**edge**: `disk` (sharp step, ringing risk in Fourier), '
    '`cosine` (cosine roll-off across `edge_width` pixels), '
    '`gauss_edge` (Gaussian roll-off), '
    '`true_gauss` (pure 2D Gaussian, `outer_radius` = sigma). '
    '**background_mode**: what value masked-out pixels get — '
    '`average` (input mean), `circumference` (perimeter mean), '
    '`external` (use `background` scalar).',
    _fn_circle_mask, manual='MA'))
