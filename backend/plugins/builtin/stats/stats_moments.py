"""Built-in node — stats.moments (FS · Statistics) — category: Statistics.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stats)

register(N('stats.moments', 'Statistics (FS)', 'Statistics',
           'image:image',
           'min:scalar,max:scalar,mean:scalar,sigma:scalar,sum:scalar,'
           'nx:scalar,ny:scalar,nz:scalar',
           [],
           'Per-image summary: minimum, maximum, mean, standard deviation, '
           'sum, plus the image dimensions **nx, ny, nz** (a 2D image reports '
           'nz=1). Handy for checking normalisation, detecting bad frames, '
           'or feeding a shape scalar into a Calculator — e.g. wire `nx` '
           'from a polar image into the calculator that converts a peak '
           'shift (columns) to degrees: `-P1 * 360 / P2`.',
           _fn_stats, manual='FS'))
