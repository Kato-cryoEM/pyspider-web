"""Built-in node — io.cp (CP · Copy image) — category: I/O.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_cp)

register(N('io.cp', 'Copy image', 'I/O', 'image:image', 'image:image', [], 'Pass-through copy. Useful as a junction point when a value has to feed several downstream nodes but you want an intermediate anchor.', _fn_cp, manual='CP'))
