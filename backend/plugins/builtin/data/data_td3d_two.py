"""Built-in node — data.td3d.two (Test · Two beads 3D) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td3d, _two_beads_3d)

register(_mk_td3d('data.td3d.two', 'Test · Two beads 3D', lambda size=64, separation_frac=0.3: _two_beads_3d(int(size), float(separation_frac)), [('separation_frac', 'float', 0.3, None, 0.05, 0.9)]))
