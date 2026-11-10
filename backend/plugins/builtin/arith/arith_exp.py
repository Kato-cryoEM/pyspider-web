"""Built-in node — arith.exp (EXP · Exponent) — category: Arithmetic.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
import numpy as np
from backend.nodes import (_un)

register(N('arith.exp', 'EXP · Exponent', 'Arithmetic', 'image:image', 'out:image', [], 'Exp(image). Careful with large positive values — output can overflow.', _un(np.exp)))
