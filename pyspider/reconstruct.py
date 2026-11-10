"""3D reconstruction from 2D projections (cryo-EM single-particle style).

This module implements the **Fourier-slice back-projection** at the heart
of SPIDER's ``BP 3F`` (``src/bp3f.f``).  The full SPIDER routine uses
Kaiser-Bessel windowed sinc interpolation for gridding accuracy; here we
keep the simpler trilinear scheme so the math is obvious.  The same
correctness invariants apply (a well-sampled set of projections recovers
the original volume).

Functions
---------
``project(volume, phi, theta, psi)``
    Forward projection of a 3D volume along a viewing direction
    specified by ZYZ Euler angles (SPIDER convention).
``bp3f(projections, eulers, volume_shape)``
    Fourier-space back-projection of a list of 2D projections.

Conventions
-----------
- Volumes are NumPy arrays of shape ``(NZ, NY, NX)`` (Z slow,
  then Y, then X).
- Projections are 2D arrays ``(NY, NX)``.
- Euler angles are in **degrees** and follow SPIDER's ZYZ convention
  ``R = Rz(psi) * Ry(theta) * Rz(phi)``; viewing direction is along
  +Z after applying ``R`` to the volume.

Reference: J. Frank "Three-Dimensional Electron Microscopy of
Macromolecular Assemblies" (Oxford, 2nd ed., 2006) §5.4.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Euler rotation matrices.


def _euler_zyz(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """ZYZ rotation matrix ``R = Rz(psi) Ry(theta) Rz(phi)`` in degrees."""
    phi, theta, psi = np.deg2rad([phi_deg, theta_deg, psi_deg])
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)
    Rz_phi = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    Ry_theta = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz_psi = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]])
    return Rz_psi @ Ry_theta @ Rz_phi


# ---------------------------------------------------------------------------
# Forward projection (sum along the rotated Z axis).


def project(
    volume: np.ndarray,
    phi_deg: float,
    theta_deg: float,
    psi_deg: float = 0.0,
) -> np.ndarray:
    """Project a 3D volume into a 2D image along the rotated +Z axis.

    Implementation: rotate the volume by ``R`` (trilinear interpolation
    via NumPy advanced indexing) and sum along the Z axis.  This is
    real-space projection, the simplest formulation.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got {vol.shape}")
    nz, ny, nx = vol.shape
    R = _euler_zyz(phi_deg, theta_deg, psi_deg).astype(np.float32)

    cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2
    # Sample coordinates in the *rotated* frame.
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    coords = np.stack([z - cz, y - cy, x - cx], axis=-1)  # (NZ, NY, NX, 3)
    rotated = coords @ R.T                                # apply R to each vector
    rz, ry, rx = rotated[..., 0] + cz, rotated[..., 1] + cy, rotated[..., 2] + cx

    sampled = _trilinear(vol, rz, ry, rx)
    # Sum along the original Z axis (= projection direction).
    return sampled.sum(axis=0).astype(np.float32)


def _trilinear(vol: np.ndarray, z: np.ndarray, y: np.ndarray,
               x: np.ndarray) -> np.ndarray:
    """Trilinear interpolation; out-of-range samples return 0.

    Allow z, y, x exactly at the upper boundary (== ``nz - 1`` etc.); we
    only need z1 when the fractional part is nonzero.
    """
    nz, ny, nx = vol.shape
    z0 = np.floor(z).astype(np.int32); z1 = z0 + 1
    y0 = np.floor(y).astype(np.int32); y1 = y0 + 1
    x0 = np.floor(x).astype(np.int32); x1 = x0 + 1
    fz = z - z0; fy = y - y0; fx = x - x0
    inside = ((z >= 0) & (y >= 0) & (x >= 0)
              & (z <= nz - 1) & (y <= ny - 1) & (x <= nx - 1))
    z0 = np.clip(z0, 0, nz - 1); z1 = np.clip(z1, 0, nz - 1)
    y0 = np.clip(y0, 0, ny - 1); y1 = np.clip(y1, 0, ny - 1)
    x0 = np.clip(x0, 0, nx - 1); x1 = np.clip(x1, 0, nx - 1)

    v000 = vol[z0, y0, x0]; v001 = vol[z0, y0, x1]
    v010 = vol[z0, y1, x0]; v011 = vol[z0, y1, x1]
    v100 = vol[z1, y0, x0]; v101 = vol[z1, y0, x1]
    v110 = vol[z1, y1, x0]; v111 = vol[z1, y1, x1]
    val = ((1 - fz) * ((1 - fy) * ((1 - fx) * v000 + fx * v001)
                       + fy * ((1 - fx) * v010 + fx * v011))
           + fz * ((1 - fy) * ((1 - fx) * v100 + fx * v101)
                   + fy * ((1 - fx) * v110 + fx * v111)))
    return np.where(inside, val, 0.0)


# ---------------------------------------------------------------------------
# Fourier-slice back-projection.


def bp3f(
    projections: np.ndarray,
    eulers: np.ndarray,
    volume_shape: tuple[int, int, int] | None = None,
    *,
    ramp_filter: bool = True,
) -> np.ndarray:
    """Filtered real-space back-projection.

    This is the classical FBP recipe: for each projection, smear it back
    along the projection direction into the 3D volume, weighting by
    ``|k|`` in Fourier space to undo the central-dip artefact of
    unfiltered back-projection.  Educational version of SPIDER ``BP 3F``;
    the Fortran original uses Fourier-domain insertion with Kaiser-Bessel
    gridding for higher fidelity at sparse angular sampling.

    Parameters
    ----------
    projections
        Array of shape ``(N, NY, NX)``.
    eulers
        Array ``(N, 3)`` of ``(phi, theta, psi)`` in degrees,
        SPIDER ZYZ convention.
    volume_shape
        Output volume shape.  Defaults to ``(NX, NX, NX)``.
    ramp_filter
        If True (default), apply the standard ``|k|`` ramp filter to
        each projection before back-projection.  Without it the
        reconstruction has a strong low-pass bias.

    Returns
    -------
    np.ndarray
        Reconstructed volume of shape ``(NZ, NY, NX)``.
    """
    projections = np.asarray(projections, dtype=np.float32)
    eulers = np.asarray(eulers, dtype=np.float32)
    if projections.ndim != 3:
        raise ValueError(f"projections must be 3D (N, NY, NX), got {projections.shape}")
    n, ny, nx = projections.shape
    if eulers.shape[0] != n or eulers.shape[1] != 3:
        raise ValueError(f"eulers must be (N, 3), got {eulers.shape}")

    nz_, ny_, nx_ = volume_shape if volume_shape is not None else (nx, nx, nx)
    cz, cy, cx = (nz_ - 1) / 2, (ny_ - 1) / 2, (nx_ - 1) / 2

    # Pre-filter projections with the 2D ramp |k| (separable in (kx, ky)).
    if ramp_filter:
        kx = np.fft.rfftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kmag = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2).astype(np.float32)
        filtered = np.empty_like(projections)
        for i in range(n):
            spec = np.fft.rfft2(projections[i]) * kmag
            filtered[i] = np.fft.irfft2(spec, s=(ny, nx)).astype(np.float32)
        projections = filtered

    # Voxel grid (centred coordinates).
    zs, ys, xs = np.mgrid[0:nz_, 0:ny_, 0:nx_].astype(np.float32)
    coords = np.stack([zs - cz, ys - cy, xs - cx], axis=-1)   # (NZ, NY, NX, 3)

    vol = np.zeros((nz_, ny_, nx_), dtype=np.float32)
    for i in range(n):
        R = _euler_zyz(*eulers[i]).astype(np.float32)
        # Map each volume voxel into the projection's local frame:
        #   v_local = R^T @ v_world
        # The projection sums along axis-0 (our "Z"), so the 2D image
        # coordinates are (y_local + cy, x_local + cx).
        local = coords @ R                          # equivalent to R^T applied to columns
        y_proj = local[..., 1] + cy
        x_proj = local[..., 2] + cx
        vol += _bilinear_2d(projections[i], y_proj, x_proj)

    return (vol / n).astype(np.float32)


def _bilinear_2d(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear sampler; out-of-image samples return 0."""
    ny, nx = img.shape
    y0 = np.floor(y).astype(np.int32); y1 = y0 + 1
    x0 = np.floor(x).astype(np.int32); x1 = x0 + 1
    fy = y - y0; fx = x - x0
    inside = (y >= 0) & (x >= 0) & (y <= ny - 1) & (x <= nx - 1)
    y0 = np.clip(y0, 0, ny - 1); y1 = np.clip(y1, 0, ny - 1)
    x0 = np.clip(x0, 0, nx - 1); x1 = np.clip(x1, 0, nx - 1)
    v = ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1])
         + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1]))
    return np.where(inside, v, 0.0).astype(np.float32)
