"""Built-in node — mask.gaussian (Gaussian mask) — category: Masking.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_gaussian_mask)

register(N('mask.gaussian', 'Gaussian mask', 'Masking',
           'image:image',
           'image:image',
           [('sigma', 'float', 0.25, None, 0.01, 2.0),
            ('background', 'float', 0.0)],
           'Gaussian radial envelope centred on the image. `sigma` is a '
           'fraction of the image half-width (0.25 = FWHM ~ 60% of the '
           'image). `background` is the value the mask fades toward at '
           'the edges. In-node preview shows the mask silhouette + a '
           'yellow horizontal cross-section so you can eye-ball the '
           'edge softness.',
           _fn_gaussian_mask))
