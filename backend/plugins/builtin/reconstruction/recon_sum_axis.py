"""Built-in node — recon.sum_axis (Sum along axis) — Reconstruction.

Collapse a 3D volume into a 2D image by summing along a chosen axis. This
is the "integration" half of a real-space projection: pair with
``geom.rotate3d`` upstream to reproduce ``recon.pj3q`` without using the
Fourier route.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, d


def _fn_sum_axis(volume, axis: str = "z"):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {vol.shape}")
    idx = {"z": 0, "y": 1, "x": 2}.get(axis)
    if idx is None:
        raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")
    return {"image": d(vol.sum(axis=idx).astype(np.float32))}


register(N(
    "recon.sum_axis", "Sum along axis", "Reconstruction",
    "volume:volume", "image:image",
    [
        ("axis", "select", "z", ["z", "y", "x"]),
    ],
    "Sum a 3D volume along the chosen axis → 2D image. Equivalent to "
    "`np.sum(volume, axis=…)`. The real-space half of a projection: feed "
    "a rotated volume in to get a projection along any direction.",
    _fn_sum_axis,
))
