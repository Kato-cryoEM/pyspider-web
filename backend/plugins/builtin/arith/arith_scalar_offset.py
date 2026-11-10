"""Built-in node — arith.scalar_offset — Arithmetic.

Shift a scalar by a constant offset (= imperative `index + k` or
`index - k`). Primarily meant for arithmetic on `flow.do.i` so that
`stack.collect` can write to `index - 1` even when the DO starts at 1,
or `stack.index` can read the previous iteration.
"""

from __future__ import annotations

from backend.plugin_api import register, N


def _fn_scalar_offset(value: float = 0.0, offset: float = 0.0):
    return {"value": float(value) + float(offset)}


register(N(
    "arith.scalar_offset", "Scalar + offset", "Arithmetic",
    "value:scalar", "value:scalar",
    [
        ("offset", "float", 0.0, None, -1e12, 1e12),
    ],
    "Add a constant `offset` to the input scalar and emit the result. "
    "Use `offset = -1` for 'previous iteration', `offset = +1` for 'next'. "
    "Combines naturally with flow.do.i to index stack.collect / stack.index.",
    _fn_scalar_offset,
))
