"""Built-in node — arith.ar_if (AR IF · Conditional arithmetic)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ar_if)

register(N(
    'arith.ar_if', 'Conditional', 'Arithmetic',
    'image:image', 'image:image',
    [Param('threshold', 'float', 0.0),
     Param('comparison', 'select', '>', ['>', '<', '>=', '<=', '==', '!=']),
     Param('then_value', 'float', 1.0),
     Param('else_value', 'float', 0.0)],
    'Per-pixel conditional. If `image[pixel] <comparison> '
    'threshold` holds, write `then_value`, else write `else_value`. Common '
    'use: quick binary mask from a scalar threshold. For arbitrary '
    'conditional formulas use `AR · Expression` with `where(cond, a, b)`.',
    _fn_ar_if, manual='AR IF'))
