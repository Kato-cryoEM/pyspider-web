"""Built-in node — arith.calc (Calculator · P= formula, dynamic inputs)."""

from __future__ import annotations

from backend.plugin_api import register, NodeSpec, Port, Param
from backend.nodes import _fn_math_expr


_MAX_INPUTS = 8

# Dynamic inputs P1..P8 — Pi becomes visible when n_inputs >= i.
_INPUTS = [
    Port(name=f"P{i}", kind="image",
         depends_on=("n_inputs", list(range(i, _MAX_INPUTS + 1))))
    for i in range(1, _MAX_INPUTS + 1)
]
# P1 is required; P2..P8 are optional so the runtime only complains when
# the formula actually references them.
for p in _INPUTS[1:]:
    p.optional = True

_OUTPUTS = [
    # Single output — an image-kind port that transparently carries a
    # 0-d scalar when the formula reduces to one number. The engine
    # (and classifyEdge) permits image → scalar so wiring into a
    # scalar param handle (RT SQ's angle_deg etc.) stays valid.
    Port(name="result", kind="image"),
]

_PARAMS = [
    Param(name="n_inputs", kind="int", default=1,
          min=1, max=_MAX_INPUTS, wireable=False),
    Param(name="formula", kind="string", default="P1",
          wireable=False,
          placeholder="P1 * 4   ·   sqrt(P1**2 + P2**2)   ·   log(1 + abs(P1))"),
]

_DOC = (
    "Calculator — evaluate a NumPy expression on wired inputs `P1`..`Pn`.\n\n"
    "Bump `n_inputs` to reveal more input handles (P2, P3, …). Reference "
    "them by NAME in the formula, e.g. `P1 * 4`, `P1 - P2`, "
    "`sqrt(P1**2 + P2**2)`, `log(1 + abs(P1))`. A leading `P=` is "
    "tolerated (`P= P1 * 4`).\n\n"
    "Inputs and outputs broadcast automatically — scalars combine "
    "element-wise with images and stacks. Both an `image` and a `scalar` "
    "output are exposed; wire whichever suits the downstream port kind.\n\n"
    "Available names: sqr, sqrt, log, log2, log10, exp, abs, sin, cos, "
    "tan, asin, acos, atan, atan2, floor, ceil, round, min, max (all "
    "element-wise), sum, mean, std, pi, e, np (full numpy)."
)

register(NodeSpec(
    id="arith.calc", label="Calculator (P= formula)", category="Arithmetic",
    inputs=_INPUTS, outputs=_OUTPUTS, params=_PARAMS,
    doc=_DOC, fn=_fn_math_expr, manual=None,
))
