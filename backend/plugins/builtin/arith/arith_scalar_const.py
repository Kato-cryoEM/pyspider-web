"""Built-in node — arith.scalar_const (Scalar constant) — Arithmetic.

Emits a single-number scalar, wireable into any numeric parameter port.
Useful for injecting fixed values into loop-aware wiring (e.g. writing
`output_stack[0] = reference` before a DO loop begins).
"""

from __future__ import annotations

from backend.plugin_api import register, N


def _fn_scalar_const(value: float = 0.0):
    return {"value": float(value)}


register(N(
    "arith.scalar_const", "Scalar constant", "Arithmetic",
    "", "value:scalar",
    [
        ("value", "float", 0.0, None, -1e12, 1e12),
    ],
    "Emit a constant numeric value as a scalar. Wire the output into any "
    "scalar parameter (e.g. stack.index.index, stack.collect.index) when "
    "you need an explicit 0 / 1 / … without arithmetic.",
    _fn_scalar_const,
))
