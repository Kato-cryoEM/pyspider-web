"""Built-in node — data.td3d.helix (Test · Helix 3D) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_helix_3d, _mk_td3d)

register(_mk_td3d('data.td3d.helix', 'Test · Helix 3D', lambda size=64, n_turns=3.0: _helix_3d(int(size), float(n_turns)), [('n_turns', 'float', 3.0, None, 0.5, 20.0)]))
