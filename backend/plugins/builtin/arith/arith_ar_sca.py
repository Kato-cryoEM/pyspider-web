"""Built-in node — arith.ar_sca (AR SCA · Rescale range)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ar_sca)

register(N(
    'arith.ar_sca', 'Rescale to [min, max]', 'Arithmetic',
    'image:image', 'image:image',
    [Param('new_min', 'float', 0.0),
     Param('new_max', 'float', 1.0)],
    'Linearly rescale so the image minimum maps to '
    '`new_min` and the maximum to `new_max`. For a constant image the '
    'output is filled with `new_min`. Handy for preview normalisation or '
    'to prepare data for Save Image (0..255 or 0..65535).',
    _fn_ar_sca, manual='AR SCA'))
