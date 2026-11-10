"""Built-in node — geom.pad (PD · Pad) — category: Geometry.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_pad)

register(N('geom.pad', 'Pad', 'Geometry', 'image:image', 'image:image', [('ny', 'int', 256), ('nx', 'int', 256), ('pad_type', 'select', 'average', ['average', 'set', 'border', 'min', 'circular']), ('background', 'float', 0.0)], "Pad type: average (A) = fill with the input's mean; set (S) = fill with the background value below; border (B) = fill with the perimeter-pixel average; min (M) = fill with the input's minimum; circular (C) = wrap the extended area around the input. background is only used when pad_type = set.", _fn_pad, manual='PD'))
