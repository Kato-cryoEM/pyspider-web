"""Built-in node — stats.histogram (HI · Histogram) — category: Statistics.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_histogram)

register(N('stats.histogram', 'Histogram', 'Statistics', 'image:image', 'image:image,min:scalar,max:scalar', [('bins', 'int', 64, None, 4, 512), ('lo', 'float', 0.0), ('hi', 'float', 0.0)], 'Compute the pixel-value histogram and render it as a bar-chart image. If both lo and hi are 0, the range auto-covers the actual min/max; otherwise the histogram is restricted to [lo, hi].', _fn_histogram, manual='HI'))
