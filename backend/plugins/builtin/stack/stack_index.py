"""Built-in node — stack.index (Stack pick) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_index)

register(N('stack.index', 'Stack pick', 'Stack', 'stack:stack', 'image:image', [('index', 'int', 0, None, 0, 100000)], 'Pull out the k-th frame of a stack as a 2D image (index is modulo the stack size). Useful for inspecting individual frames.', _fn_stack_index))
