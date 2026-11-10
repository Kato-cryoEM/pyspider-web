"""Built-in node — io.ud (UD · Read doc file) — category: I/O.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ud)

register(N('io.ud', 'Read doc file', 'I/O', '', 'v1:scalar,v2:scalar,v3:scalar,found:scalar', [('path', 'string', ''), ('key', 'int', 1)], 'Read the row whose key column matches `key` from a SPIDER text doc file and expose its first three values as scalars. found = 1 if a matching row was located, else 0.', _fn_ud, manual='UD'))
