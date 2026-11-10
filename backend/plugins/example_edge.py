"""Example plugin — Sobel edge magnitude, wired into a `Custom` category.

Drop this file (or copy it as a template) into ``backend/plugins/`` or
``~/.pyspider/plugins/``. On next restart the node appears in the palette.

Anything in this file that isn't a ``register(...)`` / ``register_reader(...)``
call is just Python — you can use scipy, cupy, or any other library you have
installed. Nothing else in the app needs to change.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from backend.plugin_api import register, N, h, d


def _fn_sobel(image, sigma: float = 1.0):
    """Gaussian-smoothed Sobel gradient magnitude."""
    a = h(image).astype(np.float32)
    if sigma > 0:
        a = ndi.gaussian_filter(a, float(sigma))
    gx = ndi.sobel(a, axis=1)
    gy = ndi.sobel(a, axis=0)
    return {"image": d(np.hypot(gx, gy).astype(np.float32))}


register(
    N("plugin.sobel", "Sobel edge (plugin)", "Custom",
      "image:image", "image:image",
      [("sigma", "float", 1.0, None, 0.0, 20.0)],
      "Example plugin: Gaussian-smoothed Sobel gradient magnitude. "
      "Delivered from backend/plugins/example_edge.py — read that file to "
      "see how to write your own.",
      _fn_sobel),
)
