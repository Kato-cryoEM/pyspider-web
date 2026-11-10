"""Built-in node — stack.mean (AS R · Stack mean) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_average)

register(N('stack.mean', 'Stack mean', 'Stack', 'stack:stack', 'image:image', [], "Average of a 3D image stack along the frame axis. Simplest 'reference image' for a set of aligned particles.", _fn_stack_average, manual='AS R'))
