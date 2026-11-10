"""Built-in node — data.checker (Test · Checkerboard) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_mk_td, _td)

register(_mk_td('data.checker', 'Test · Checkerboard', lambda size=256, square_size=16: _td.checkerboard(int(size), int(square_size)), [('square_size', 'int', 16, None, 2, 64)]))
