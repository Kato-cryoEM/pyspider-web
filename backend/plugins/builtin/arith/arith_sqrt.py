"""Built-in node — arith.sqrt (SQRT · Square-root) — category: Arithmetic.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
import numpy as np
from backend.nodes import (_un)

register(N('arith.sqrt', 'Square-root', 'Arithmetic', 'image:image', 'out:image', [], 'Sqrt(|image|). Compresses bright/dark dynamic range.', _un(lambda a: np.sqrt(np.abs(a))), manual='SQRT'))
