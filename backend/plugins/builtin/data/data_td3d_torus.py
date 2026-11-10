"""Built-in node — data.td3d.torus (Test · Torus 3D) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td3d, _torus_3d)

register(_mk_td3d('data.td3d.torus', 'Test · Torus 3D', lambda size=64, R_frac=0.28, r_frac=0.08: _torus_3d(int(size), float(R_frac), float(r_frac)), [('R_frac', 'float', 0.28, None, 0.05, 0.9), ('r_frac', 'float', 0.08, None, 0.01, 0.5)]))
