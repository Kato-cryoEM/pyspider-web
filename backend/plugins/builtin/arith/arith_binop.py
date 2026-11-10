"""Built-in node — arith.binop (element-wise binary op) — Arithmetic.

Consolidates the four former nodes `arith.add`, `arith.sub`, `arith.mul`,
`arith.div` into one node with an `op` dropdown. All four shared the exact
same I/O signature (two image inputs → one image output); only the lambda
differed. Collapsing them lets a user flip between + / − / × / ÷ by
changing one parameter instead of replacing the node and rewiring.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param, d
from backend.nodes import h, _promote


_OPS = ["add", "sub", "mul", "div"]

# Human-readable symbols used both in the docstring and (optionally)
# in future UI label expansion. Kept out of the Param itself so the
# dropdown values stay machine-friendly (`add`, not `+`).
_SYMBOLS = {"add": "+", "sub": "−", "mul": "×", "div": "÷"}


def _fn_binop(a, b, op: str = "add"):
    ah, bh = h(a), h(b)
    dt = _promote(ah, bh)
    x = np.asarray(ah, dtype=dt)
    y = np.asarray(bh, dtype=dt)
    if op == "add":
        out = x + y
    elif op == "sub":
        out = x - y
    elif op == "mul":
        out = x * y
    elif op == "div":
        # Match the former arith.div behaviour: 0/0 → 0 instead of NaN.
        out = np.divide(x, y, out=np.zeros_like(x), where=(y != 0))
    else:
        raise ValueError(f"unknown op: {op!r}. Known: {_OPS}")
    return {"out": d(out)}


_DOC = (
    "Element-wise binary operation on two images of the same shape:\n\n"
    "  • `add`  — pixelwise sum (a + b). Classic use: averaging two images.\n"
    "  • `sub`  — pixelwise difference (a − b). Useful for highlighting "
    "where two processings disagree (diff image).\n"
    "  • `mul`  — pixelwise product (a × b). Standard way to apply a mask: "
    "wire image into `a`, mask into `b`.\n"
    "  • `div`  — pixelwise quotient (a ÷ b). 0/0 is replaced by 0 to "
    "avoid NaN propagation.\n\n"
    "Mixed int / float / complex inputs are promoted to the common dtype "
    "before the operation (same convention as NumPy)."
)


register(N(
    "arith.binop", "Arithmetic · a ∘ b", "Arithmetic",
    "a:image,b:image", "out:image",
    [Param("op", "select", "add", _OPS)],
    _DOC, _fn_binop,
))
