"""Built-in node — arith.add_stack (ADD · Sum images/stack)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_add)

register(N(
    'arith.add_stack', 'Sum images', 'Arithmetic',
    'stack:stack', 'image:image', [],
    'Sum every frame of a stack into one image. To combine '
    '2 – 4 individual images, feed them through `Stack · Stack build` '
    'first. For just two operands use `AD · Add` or `ADD 2 · Sum two`.',
    _fn_add, manual='ADD'))
