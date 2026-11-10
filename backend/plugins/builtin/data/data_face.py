"""Built-in node — data.face (Test · Face) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td, _td)

register(_mk_td('data.face', 'Test · Face', lambda size=128, eyes='open': _td.face(int(size), eyes), [('eyes', 'select', 'open', ['open', 'closed'])]))
