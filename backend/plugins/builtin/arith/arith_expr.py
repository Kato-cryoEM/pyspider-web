"""Built-in node — arith.expr (AR · Expression) — category: Arithmetic.

SPIDER-conforming AR: a per-pixel formula evaluator with P1..P4 image
inputs and [x] / [y] coordinate references. Matches the semantics of the
real SPIDER AR command (docs/man/ar.html).
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ar_expr)

register(N(
    'arith.expr', 'Expression', 'Arithmetic',
    'p1:image,p2:image?,p3:image?,p4:image?', 'image:image',
    [Param('formula', 'string', 'P1',
           placeholder='e.g.  (P1 - 2.0) / 2.0 + SIN(P1)   ·   P1 * P2   ·   where(P1 > 0.5, P1, 0)')],
    'Evaluate a per-pixel arithmetic expression. '
    'Variables: **P1** = required first input; **P2, P3, P4** = optional '
    'extra images; **[x], [y]** = pixel coordinates. '
    'Supported functions (case-insensitive): SIN, COS, TAN, ASIN, ACOS, '
    'ATAN, ATAN2, SINH, COSH, TANH, EXP, LOG, LOG10, LOG2, SQRT, ABS, '
    'SIGN, FLOOR, CEIL, ROUND, MIN, MAX, CLIP, WHERE. '
    'Constants: PI, E. Operators: `+ - * / % **` and `^` (alias for `**`), '
    'comparison `< > == != <= >=`, boolean `and or not`, ternary '
    '`a if cond else b`. '
    'Examples from the SPIDER manual: `(P1 - 2.0) / 2.0 + SIN(P1 + 2)`, '
    '`(P1 / 512.) + 0.`, `EXP(LOG(P1 / 512.) + [x] + [y])`. '
    'Expressions are AST-validated (no attribute access, imports, or '
    'arbitrary calls), so unsafe code is rejected before evaluation.',
    _fn_ar_expr, manual='AR'))
