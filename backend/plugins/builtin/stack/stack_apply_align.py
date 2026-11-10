"""Built-in node — stack.apply_align (Apply per-frame alignment CSV) — category: Stack."""

from __future__ import annotations

from backend.plugin_api import register, N
from backend.nodes import _fn_stack_apply_align

register(N(
    'stack.apply_align', 'Apply per-frame align (rot+shift+mirror)', 'Stack',
    'stack:stack,eulers_csv:string',
    'stack:stack',
    [
        ('cval', 'float', 0.0),
        ('interpolation', 'select', 'cubic',
         ['nearest', 'linear', 'cubic', 'quintic']),
    ],
    'Apply per-frame alignment (rotation `phi` + shift `dy/dx` + `mirror`) '
    'to each frame of a stack using the CSV format produced by AP SHC / '
    'AP SR. One row per frame in the form `phi,0,0  # dy=<v> dx=<v> '
    'mirror=<0|1>`. Row count must equal the stack frame count. Use this '
    'to transfer alignment from a decimated / masked stack to a full-'
    'resolution or differently-preprocessed copy of the same particles.',
    _fn_stack_apply_align))
