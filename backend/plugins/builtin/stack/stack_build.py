"""Built-in node — stack.build (Stack build (2–4)) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_stack_build)

register(N('stack.build', 'Stack build (2–4)', 'Stack', 'a:image,b:image,c:image?,d_:image?', 'stack:stack', [], 'Combine 2 – 4 images into a stack along a new leading axis. Feed the resulting stack to AS S / AP SHC / BP 3F / classification.', _fn_stack_build))
