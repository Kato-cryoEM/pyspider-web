"""Built-in node — io.fi (FI · File info) — category: I/O.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_fi)

register(N('io.fi', 'File info', 'I/O', '', 'nx:scalar,ny:scalar,nz:scalar,fmax:scalar,fmin:scalar,av:scalar,sig:scalar', [('path', 'string', '')], "Read the SPIDER header of a file and emit its dimensions plus stored min / max / mean / sigma statistics — useful for driving downstream parameters (e.g. wire av into MA's background).", _fn_fi, manual='FI'))
