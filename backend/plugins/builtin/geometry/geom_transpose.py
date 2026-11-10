"""Built-in node — geom.transpose (Transpose) — category: Geometry.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_transpose)

register(N('geom.transpose', 'Transpose', 'Geometry', 'image:image', 'image:image', [], 'Swap axes (image.T).', _fn_transpose, manual='TP'))
