"""Built-in node — arith.sq (SQ · Square) — category: Arithmetic.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
import numpy as np
from backend.nodes import (_un)

register(N('arith.sq', 'Square', 'Arithmetic', 'image:image', 'out:image', [], 'Element-wise square (image²).', _un(np.square), manual='SQ'))
