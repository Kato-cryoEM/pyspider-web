"""Built-in node — data.td3d.shepp (Test · Shepp-Logan 3D) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td3d)

register(_mk_td3d('data.td3d.shepp', 'Test · Shepp-Logan 3D', lambda size=64: _shepp_logan_3d(int(size))))
