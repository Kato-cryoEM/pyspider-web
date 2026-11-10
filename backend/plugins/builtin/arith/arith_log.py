"""Built-in node — arith.log (LN · Log) — category: Arithmetic.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
import numpy as np
from backend.nodes import (_un)

register(N('arith.log', 'Log', 'Arithmetic', 'image:image', 'out:image', [], 'Log(1 + |image|). Numerically stable for zeros and negatives.', _un(lambda a: np.log1p(np.abs(a)))))
