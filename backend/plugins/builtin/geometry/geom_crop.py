"""Built-in node — geom.crop (Window / crop) — category: Geometry.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_crop)

register(N(
    'geom.crop', 'Window (crop)', 'Geometry',
    'image:image', 'image:image',
    [
        Param('center', 'bool', True),
        Param('y', 'int', 0, depends_on=('center', False)),
        Param('x', 'int', 0, depends_on=('center', False)),
        Param('ny', 'int', 64),
        Param('nx', 'int', 64),
    ],
    'Extract a sub-window of size (ny, nx). '
    '**center = ON (default)**: the crop is auto-centred on the array '
    'centre — `y`/`x` are ignored and hidden. This is what you want for '
    '**Fourier cropping** (FT → Window → IFT) because FT output has DC at '
    'the centre; a corner-anchored crop would shift DC off-centre and '
    'produce a phase ramp / wrap-around in the inverse. It is also the '
    'natural "smaller box around the object" behaviour for real-space '
    'images. '
    '**center = OFF**: the crop starts at pixel (y, x) — the traditional '
    'SPIDER WI corner-anchored crop. Use this when you want to sample a '
    'specific off-centre region of a real-space image.',
    _fn_crop, manual='WI'))
