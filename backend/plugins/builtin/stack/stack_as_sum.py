"""Built-in node — stack.as_sum (AS · Sum stack + variance) — Stack."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_as_sum)

register(N(
    'stack.as_sum', 'Sum stack + variance', 'Stack',
    'stack:stack', 'sum:image,variance:image,mean_of_variance:scalar',
    [Param('mode', 'select', 'sum_and_variance',
           ['sum_and_variance', 'sum', 'variance'])],
    'Sum every frame of the stack (per-pixel sum) and emit per-'
    'pixel variance across frames. `mean_of_variance` is the average of the '
    'variance image (SPIDER AS `avgoff` register). See AS R for the '
    'average-instead-of-sum variant, or AS S for the two-image case.',
    _fn_as_sum, manual='AS'))
