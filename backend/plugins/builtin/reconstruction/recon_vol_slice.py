"""Built-in node — recon.vol_slice (Volume slice) — category: Reconstruction.

Pull a single 2D plane out of a 3D volume along a chosen axis. The default
``index = -1`` means "the geometric centre" (``N // 2``). Useful for inspecting
density maps slice-by-slice and for staging the input of a Fourier-slice
demonstration (compare ``recon.central_slice`` for its k-space cousin).
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, d


def _fn_vol_slice(volume, axis: str = "z", index: int = -1):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {vol.shape}")
    nz, ny, nx = vol.shape
    if axis == "z":
        n = nz; sl = lambda i: vol[i, :, :]
    elif axis == "y":
        n = ny; sl = lambda i: vol[:, i, :]
    elif axis == "x":
        n = nx; sl = lambda i: vol[:, :, i]
    else:
        raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")
    i = int(index)
    if i < 0:
        i = n // 2
    i = max(0, min(n - 1, i))
    return {"image": d(sl(i).astype(np.float32))}


register(N(
    "recon.vol_slice", "Volume slice", "Reconstruction",
    "volume:volume", "image:image",
    [
        ("axis", "select", "z", ["z", "y", "x"]),
        ("index", "int", -1, None, -1, 10000),
    ],
    "Extract one 2D plane from a 3D volume along the chosen axis. "
    "`index = -1` picks the centre slice (N // 2); otherwise the given "
    "integer indexes into the chosen axis.",
    _fn_vol_slice,
))
