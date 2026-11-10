"""Built-in node — geom.po (PO · Polar transform) — category: Geometry.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_po)

register(N('geom.po', 'Polar transform', 'Geometry', 'image:image', 'image:image', [('inner_radius', 'int', 1, None, 0, 4096), ('outer_radius', 'int', 0, None, 0, 4096), ('center_x', 'float', -1.0), ('center_y', 'float', -1.0)], 'Convert rectangular image to polar-coordinate representation. Rows range from `inner_radius` (row 0) to `outer_radius` (last row); columns are angular samples along each ring (length ≈ 2π · outer_radius). Sweep starts at +x (horizontal right of centre) and goes clockwise. **inner_radius = 1** by default; **outer_radius = 0 means auto** = half of the shorter image side. center_x / center_y = -1 auto-uses the image centre. Useful pre-processing for rotational alignment / autocorrelation on rings.', _fn_po, manual='PO'))
