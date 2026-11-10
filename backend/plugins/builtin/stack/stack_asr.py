"""Built-in node — stack.asr (AS R · Average stack + variance) — Stack."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_asr)

register(N(
    'stack.asr', 'Average stack + variance', 'Stack',
    'stack:stack',
    'average:image,variance:image,average_a:image,average_b:image,'
    'n_a:scalar,n_b:scalar',
    [Param('mode', 'select', 'average_and_variance',
           ['average_and_variance', 'average', 'variance', 'two_subsets']),
     Param('seed', 'int', 42, depends_on=('mode', 'two_subsets'))],
    'Average of a stack + per-pixel variance across frames. '
    'The `two_subsets` mode randomly splits the stack in half and returns '
    'two subset averages (for FSC-style comparison). `seed` controls the '
    'split (only used in `two_subsets`).',
    _fn_asr, manual='AS R'))
