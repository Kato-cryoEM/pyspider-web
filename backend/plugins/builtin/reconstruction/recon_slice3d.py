"""Built-in node — ft.slice3d (3D → 2D central slice) — Fourier.

Pick a 2D plane out of a 3D array (real-space volume OR 3D Fourier). ``index
= -1`` means the geometric centre; ``tilt_deg`` lets you rotate the slice
plane away from axis-aligned. The Inspector draws a live diagram of the
slice position above the parameters so you don't need to run the graph to
see what plane you're about to extract.

Pairs with ``ft.forward3d``:

  volume → ft.forward3d → ft.slice3d(axis=y) → ft.inverse

reproduces the real-space projection ``sum_y V`` by the central-slice
theorem.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N


def _trilinear_complex(F: np.ndarray, z: np.ndarray, y: np.ndarray, x: np.ndarray
                        ) -> np.ndarray:
    """Trilinear interpolation of a 3D (possibly complex) array; OOB → 0."""
    nz, ny, nx = F.shape
    z0 = np.floor(z).astype(np.int32); z1 = z0 + 1
    y0 = np.floor(y).astype(np.int32); y1 = y0 + 1
    x0 = np.floor(x).astype(np.int32); x1 = x0 + 1
    fz = z - z0; fy = y - y0; fx = x - x0
    inside = ((z >= 0) & (y >= 0) & (x >= 0)
              & (z <= nz - 1) & (y <= ny - 1) & (x <= nx - 1))
    z0 = np.clip(z0, 0, nz - 1); z1 = np.clip(z1, 0, nz - 1)
    y0 = np.clip(y0, 0, ny - 1); y1 = np.clip(y1, 0, ny - 1)
    x0 = np.clip(x0, 0, nx - 1); x1 = np.clip(x1, 0, nx - 1)
    v000 = F[z0, y0, x0]; v001 = F[z0, y0, x1]
    v010 = F[z0, y1, x0]; v011 = F[z0, y1, x1]
    v100 = F[z1, y0, x0]; v101 = F[z1, y0, x1]
    v110 = F[z1, y1, x0]; v111 = F[z1, y1, x1]
    val = ((1 - fz) * ((1 - fy) * ((1 - fx) * v000 + fx * v001)
                       + fy * ((1 - fx) * v010 + fx * v011))
           + fz * ((1 - fy) * ((1 - fx) * v100 + fx * v101)
                   + fy * ((1 - fx) * v110 + fx * v111)))
    return np.where(inside, val, 0.0)


def _fn_slice3d(volume, axis: str = "z", index: int = -1,
                 tilt_deg: float = 0.0):
    F = np.asarray(volume)
    if F.ndim != 3:
        raise ValueError(f"input must be 3D, got {F.shape}")
    nz, ny, nx = F.shape

    # Fast axis-aligned path for the common tilt=0 case (no interpolation).
    if abs(float(tilt_deg)) < 1e-9:
        if axis == "z":
            n = nz; take = lambda i: F[i, :, :]
        elif axis == "y":
            n = ny; take = lambda i: F[:, i, :]
        elif axis == "x":
            n = nx; take = lambda i: F[:, :, i]
        else:
            raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")
        i = int(index)
        if i < 0:
            i = n // 2
        i = max(0, min(n - 1, i))
        # Keep natural dtype: a real volume stays float32, a 3D fourier stays
        # complex128. Downstream nodes (ft.inverse / ft.amplitude) accept both.
        return {"image": take(i)}

    # Tilted path: trilinear sampling of an oblique plane through centre.
    θ = float(np.deg2rad(float(tilt_deg)))
    cθ, sθ = float(np.cos(θ)), float(np.sin(θ))
    # Basis vectors expressed in (dz, dy, dx) order to match array indexing.
    if axis == "z":
        out_h, out_w = ny, nx
        u_row = np.array([-sθ, cθ, 0.0])
        u_col = np.array([0.0, 0.0, 1.0])
    elif axis == "y":
        out_h, out_w = nz, nx
        u_row = np.array([cθ, -sθ, 0.0])
        u_col = np.array([0.0, 0.0, 1.0])
    elif axis == "x":
        out_h, out_w = nz, ny
        u_row = np.array([cθ, 0.0, -sθ])
        u_col = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")

    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    i, j = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    i -= (out_h - 1) / 2.0
    j -= (out_w - 1) / 2.0
    dz = cz + i * u_row[0] + j * u_col[0]
    dy = cy + i * u_row[1] + j * u_col[1]
    dx = cx + i * u_row[2] + j * u_col[2]
    sampled = _trilinear_complex(F, dz, dy, dx)
    return {"image": sampled}


register(N(
    "recon.slice3d", "Slice (3D → 2D plane)", "Reconstruction",
    # Any 3D array is OK as input (real-space volume OR 3D Fourier). Output
    # is a 2D image whose dtype follows the input (float32 for real volumes,
    # complex128 for 3D Fourier). ft.inverse / ft.amplitude accept either.
    "volume:volume", "image:image",
    [
        ("axis", "select", "z", ["z", "y", "x"]),
        ("index", "int", -1, None, -1, 10000),
        ("tilt_deg", "float", 0.0, None, -180.0, 180.0),
    ],
    "Extract a 2D plane from a 3D array (real-space volume or 3D Fourier). "
    "`axis` picks the primary direction (perpendicular to the slice when "
    "tilt=0). `index = -1` picks the centre (DC plane for Fourier); other "
    "values index into the chosen axis. `tilt_deg` tilts the slice away "
    "from axis-aligned by rotating the normal around a secondary axis "
    "(x̂ for axis='z' or 'y'; ŷ for 'x'). The Inspector draws a live "
    "diagram of the slice position so you can preview before running.",
    _fn_slice3d,
))
