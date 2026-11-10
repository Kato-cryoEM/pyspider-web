"""Built-in node — geom.rotate3d (Rotate volume) — Geometry.

Rotate a 3D volume by SPIDER ZYZ Euler angles (phi, theta, psi, degrees)
about its geometric centre. Trilinear interpolation; the output has the
same shape as the input, zero-padded where the rotated volume falls
outside the original box.

This is the "rotation" step of ``recon.pj3q`` extracted on its own, so
pairing with ``recon.sum_axis(axis=z)`` reproduces pj3q by hand.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param, d
from pyspider.reconstruct import _euler_zyz, _trilinear


def _fn_rotate3d(volume, phi: float = 0.0, theta: float = 0.0, psi: float = 0.0):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {vol.shape}")
    R = _euler_zyz(float(phi), float(theta), float(psi)).astype(np.float32)
    nz, ny, nx = vol.shape
    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    coords = np.stack([z - cz, y - cy, x - cx], axis=-1)
    rotated = coords @ R.T
    rz = rotated[..., 0] + cz
    ry = rotated[..., 1] + cy
    rx = rotated[..., 2] + cx
    out = _trilinear(vol, rz, ry, rx).astype(np.float32)
    return {"volume": d(out)}


register(N(
    "geom.rotate3d", "Rotate volume (ZYZ)", "Geometry",
    "volume:volume", "volume:volume",
    # step=1 so the Inspector's up/down spinner nudges by 1° — angles read
    # naturally in whole degrees here, finer resolution is rarely useful.
    [
        Param("phi",   "float", 0.0, min=-360.0, max=360.0, step=1.0),
        Param("theta", "float", 0.0, min=-360.0, max=360.0, step=1.0),
        Param("psi",   "float", 0.0, min=-360.0, max=360.0, step=1.0),
    ],
    "Rotate a volume by SPIDER ZYZ Euler angles (phi, theta, psi, deg) "
    "about its centre. Combine with recon.sum_axis(axis=z) to reproduce "
    "recon.pj3q by hand — useful when you want to see the rotated volume "
    "or interpose a filter between rotation and summation.",
    _fn_rotate3d,
))
