"""Built-in node — stack.var (Stack variance) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_variance)

register(N('stack.var', 'Stack variance', 'Stack', 'stack:stack', 'image:image', [], 'Per-pixel variance across all frames of a stack. Bright pixels mark spots where frames disagree — useful for detecting heterogeneity or bad frames.', _fn_stack_variance))
