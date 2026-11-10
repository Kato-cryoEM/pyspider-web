"""Built-in node — data.coins (Test · Coins) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td, _td)

register(_mk_td('data.coins', 'Test · Coins', lambda size=256: _td.coins(int(size))))
