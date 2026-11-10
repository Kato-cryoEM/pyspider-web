"""Built-in node — stack.ass (AS S · Average + variance) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ass)

register(N('stack.ass', 'Average + variance', 'Stack', 'stack:stack', 'average:image,variance:image', [('mode', 'select', 'average_and_variance', ['average', 'variance', 'average_and_variance'])], 'Averages a stack and (optionally) computes per-pixel variance. Choose average, variance, or both (A / V / A V menu).', _fn_ass, manual='AS S'))
