"""Built-in node — geom.rotate3d (Rotate volume) — Geometry.

Rotate a 3D volume by three Euler angles (phi, theta, psi, degrees) about
its geometric centre. The `convention` dropdown picks the rotation
sequence — defaulting to **ZYZ (intrinsic)**, which is what every modern
cryo-EM package uses (SPIDER, RELION, cryoSPARC, XMIPP, FREALIGN, EMAN2,
Scipion). The other 11 conventions are provided for interop with tools
outside cryo-EM (aerospace, robotics, 3D graphics, generic scipy code).

All 12 options are intrinsic — each successive rotation is about the
body-fixed axis that just rotated, which matches scipy's lowercase
`Rotation.from_euler` convention.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from backend.plugin_api import register, N, Param, d
from pyspider.reconstruct import _trilinear


# Full 12 intrinsic Euler-angle sequences supported by scipy. Proper Euler
# angles (same axis for the first and third) + Tait-Bryan (three distinct
# axes). Order chosen so cryo-EM users land on the right one immediately.
_CONVENTIONS = [
    "zyz",  # SPIDER / RELION / cryoSPARC / XMIPP / FREALIGN / EMAN2
    "zxz",  # classical mechanics (Goldstein)
    "xyz",  # 3D graphics
    "xzy",
    "yxz",
    "yzx",
    "zyx",  # aerospace / robotics (yaw-pitch-roll)
    "zxy",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
]


def _fn_rotate3d(volume, phi: float = 0.0, theta: float = 0.0,
                  psi: float = 0.0, convention: str = "zyz"):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {vol.shape}")
    conv = convention.strip().lower()
    if conv not in _CONVENTIONS:
        raise ValueError(
            f"convention {convention!r} is not one of {_CONVENTIONS}")
    R = _Rotation.from_euler(
        conv, [float(phi), float(theta), float(psi)], degrees=True
    ).as_matrix().astype(np.float32)
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


# Human-readable labels for the dropdown — annotates which conventions are
# the cryo-EM standard vs. useful for interop with other fields. The
# frontend renders these verbatim.
_CHOICES = [
    "zyz (SPIDER / RELION / cryoSPARC)",
    "zxz (classical, Goldstein)",
    "xyz (3D graphics)",
    "xzy",
    "yxz",
    "yzx",
    "zyx (aerospace yaw-pitch-roll)",
    "zxy",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
]


def _fn_rotate3d_dispatch(volume, phi=0.0, theta=0.0, psi=0.0,
                           convention="zyz (SPIDER / RELION / cryoSPARC)"):
    """Thin dispatcher: strip the parenthesised label and forward."""
    conv = str(convention).split(" ", 1)[0].strip().lower()
    return _fn_rotate3d(volume, phi, theta, psi, conv)


register(N(
    "geom.rotate3d", "Rotate volume (Euler)", "Geometry",
    "volume:volume", "volume:volume",
    [
        Param("phi",   "float", 0.0, min=-360.0, max=360.0, step=1.0),
        Param("theta", "float", 0.0, min=-360.0, max=360.0, step=1.0),
        Param("psi",   "float", 0.0, min=-360.0, max=360.0, step=1.0),
        Param("convention", "select",
              "zyz (SPIDER / RELION / cryoSPARC)", _CHOICES),
    ],
    "Rotate a volume by three Euler angles (phi, theta, psi, degrees) "
    "about its centre. The `convention` dropdown picks the rotation "
    "sequence — all are intrinsic (each rotation about the new body "
    "axis, matching scipy's lowercase `Rotation.from_euler`).\n\n"
    "- **zyz (SPIDER / RELION / cryoSPARC)**: cryo-EM standard. Also used "
    "by XMIPP, FREALIGN, EMAN2, Scipion — match this if you're round-"
    "tripping angles between pyspider and a published reconstruction.\n"
    "- **zxz (classical, Goldstein)**: textbook rigid-body mechanics.\n"
    "- **xyz (3D graphics)**: pitch-yaw-roll in many graphics pipelines.\n"
    "- **zyx (aerospace yaw-pitch-roll)**: flight dynamics convention.\n"
    "- Other Tait-Bryan (xzy, yxz, yzx, zxy) and proper-Euler (xyx, xzx, "
    "yxy, yzy) sequences are included for interop with generic scipy / "
    "robotics code.\n\n"
    "Combine with `recon.sum_axis(axis=z)` to reproduce `recon.pj3q` by "
    "hand — useful when you want to see the rotated volume or interpose "
    "a filter between rotation and summation.",
    _fn_rotate3d_dispatch,
))
