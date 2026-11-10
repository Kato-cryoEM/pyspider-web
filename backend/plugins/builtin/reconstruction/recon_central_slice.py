"""Built-in node — recon.central_slice (Central Fourier slice) — Reconstruction.

Compute ``fftshift(fftn(volume))`` and return the central 2D plane
perpendicular to the chosen axis. The output is a 2D ``fourier`` object
with DC at ``(ny // 2, nx // 2)``, i.e. the same convention emitted by
``ft.forward`` — so wiring the result into ``ft.amplitude`` or ``ft.inverse``
works without any manual shift.

Projection-slice theorem: this central slice is exactly the 2D Fourier
transform of the real-space projection of the volume along the chosen
axis. ``recon.central_slice → ft.inverse`` therefore reproduces the
projection you'd get from ``recon.pj3q`` — try both to verify.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N


def _fn_central_slice(volume, axis: str = "z"):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {vol.shape}")
    F3 = np.fft.fftshift(np.fft.fftn(vol))
    nz, ny, nx = vol.shape
    if axis == "z":
        slc = F3[nz // 2, :, :]
    elif axis == "y":
        slc = F3[:, ny // 2, :]
    elif axis == "x":
        slc = F3[:, :, nx // 2]
    else:
        raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")
    # fftshift was applied to all 3 axes, so the extracted 2D plane already
    # has DC at (ny // 2, nx // 2) — compatible with ft.inverse / ft.amplitude.
    return {"fourier": slc.astype(np.complex128)}


register(N(
    "recon.central_slice", "Central Fourier slice", "Reconstruction",
    "volume:volume", "fourier:fourier",
    [
        ("axis", "select", "z", ["z", "y", "x"]),
    ],
    "Central slice of the 3D FFT of the volume, perpendicular to the "
    "chosen axis. By the projection-slice theorem this equals the 2D FFT "
    "of a real-space projection of the volume along the same axis — feed "
    "this into ft.inverse and you get the projection back.",
    _fn_central_slice,
))
