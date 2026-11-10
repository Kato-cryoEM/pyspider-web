"""Built-in node — arith.add2 (ADD 2 · Sum two images)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_add2)

register(N(
    'arith.add2', 'Sum two', 'Arithmetic',
    'a:image,b:image', 'image:image', [],
    'Element-wise sum of two images / volumes / Fourier '
    'arrays: `P3 = P1 + P2`. Equivalent to `AR · Expression` with `P1 + P2` '
    'or to `AD · Add` — provided as its own node under the SPIDER-canonical '
    'name.',
    _fn_add2, manual='ADD 2'))
