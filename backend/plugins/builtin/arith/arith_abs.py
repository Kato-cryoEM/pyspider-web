"""Built-in node — arith.abs (ABS · Absolute) — category: Arithmetic.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
import numpy as np
from backend.nodes import (_un)

register(N('arith.abs', 'ABS · Absolute', 'Arithmetic', 'image:image', 'out:image', [], '|image|. Turns signed data into magnitude.', _un(np.abs)))
