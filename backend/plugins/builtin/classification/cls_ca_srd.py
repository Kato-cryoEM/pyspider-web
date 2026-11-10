"""Built-in node — cls.ca_srd (CA SRD · Differential eigen-image)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ca_srd)

register(N(
    'cls.ca_srd', 'Differential eigen-image', 'Classification',
    'eigenimages:stack,eigenvalues:stack', 'image:image',
    [Param('factor_index', 'int', 0, None, 0, 256)],
    'Differential (eigen) image for a single factor. Returns '
    '`eigenimage[factor_index] * sqrt(eigenvalue)` so magnitudes across '
    'factors are comparable. Useful for inspecting the variation direction '
    'associated with the k-th principal component.',
    _fn_ca_srd, manual='CA SRD'))
