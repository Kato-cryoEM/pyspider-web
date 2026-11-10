"""Built-in node — data.siemens (Test · Siemens Star) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td, _td)

register(_mk_td('data.siemens', 'Test · Siemens Star', lambda size=256, spokes=32: _td.siemens_star(int(size), int(spokes)), [('spokes', 'int', 32, None, 4, 128)]))
