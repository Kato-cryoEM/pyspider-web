"""Built-in node — arith.addr (ADD R · Ratio-weighted sum)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_addr)

register(N(
    'arith.addr', 'Ratio-weighted sum', 'Arithmetic',
    'a:image,b:image', 'image:image',
    [Param('factor_a', 'float', 1.0),
     Param('factor_b', 'float', 1.0),
     Param('eps', 'float', 1e-12)],
    'Ratio-weighted element-wise sum: '
    '`P3 = P1 * factor_a + (P1 / P2) * factor_b`. `eps` is a small floor '
    'that replaces zero divisors on `P2` to keep the output finite (SPIDER '
    'silently produces inf there).',
    _fn_addr, manual='ADD R'))
