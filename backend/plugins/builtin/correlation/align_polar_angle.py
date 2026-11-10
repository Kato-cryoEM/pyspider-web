"""Built-in node — align.polar_angle (Polar peak → rotation angle) — category: Correlation."""

from __future__ import annotations

from backend.plugin_api import register, N
from backend.nodes import _fn_polar_angle

register(N(
    'align.polar_angle', 'Polar peak → angle (°)', 'Correlation',
    'polar_map:image,dx:scalar',
    'angle_deg:scalar,n_cols:scalar',
    [('sign', 'select', 'invert', ['invert', 'same'])],
    'Convert a polar-CC peak\'s `dx` (pixel-columns from centre) into a '
    'rotation angle in degrees, using the polar image\'s column count '
    '(one column per angular sample around 360°). `sign=invert` (default) '
    'flips the sign so the returned `angle_deg` is what you feed to RT / '
    'RT SQ to rotate the QUERY image BACK into alignment with the '
    'reference. Wire the reference or query polar image into `polar_map` '
    '(any polar image works — only the width matters).',
    _fn_polar_angle))
