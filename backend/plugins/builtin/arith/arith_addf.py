"""Built-in node — arith.addf (ADD F · Weighted sum)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_addf)

register(N(
    'arith.addf', 'Weighted sum', 'Arithmetic',
    'a:image,b:image', 'image:image',
    [Param('factor_a', 'float', 1.0),
     Param('factor_b', 'float', 1.0)],
    'Weighted element-wise sum: '
    '`P3 = P1 * factor_a + P2 * factor_b`. Both factors are wireable so you '
    'can drive them from other nodes (e.g. reciprocal of FS · sigma for '
    'variance-weighted averaging).',
    _fn_addf, manual='ADD F'))
