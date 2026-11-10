"""Random generators for projections, rotations, and synthetic particle stacks.

The functions here produce the kind of input a textbook alignment or
reconstruction demo needs: a 3D blob volume, a stack of random projections
from random viewing directions, or a stack of random in-plane rotations
of a 2D template.  Everything is pure NumPy and returns ``float32``
arrays so the output drops straight into ``pyspider.align`` /
``pyspider.reconstruct`` / ``pyspider.classify``.

All random streams are seeded via ``np.random.default_rng(seed)`` so
results are reproducible by default.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import reconstruct


# ---------------------------------------------------------------------------
# 3D volume: random Gaussian-blob particle.


def random_volume(
    N: int = 64,
    n_blobs: int = 10,
    seed: int = 0,
    core_contrast: float = 0.9,
    soft_edge: bool = True,
) -> np.ndarray:
    """A random blob-shaped 3D particle sitting on a zero background.

    Built from one central Gaussian "body" plus ``n_blobs`` smaller
    off-centre Gaussian sub-domains.  Values are normalised to ``[0, 1]``.
    Shape: ``(N, N, N)`` with axis order ``(Z, Y, X)`` matching
    :func:`pyspider.reconstruct.project`.

    Parameters
    ----------
    N
        Cube edge length.
    n_blobs
        Number of off-centre Gaussian sub-domains added to the central body.
    seed
        RNG seed; same seed → same volume.
    core_contrast
        Peak density of the central body before normalisation (0-1 scale).
    soft_edge
        If True, mask the volume with a smooth spherical envelope so the
        particle does not touch the box edges (recommended when the volume
        will be fed to projection / reconstruction).
    """
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
    c = (N - 1) / 2.0

    vol = np.zeros((N, N, N), dtype=np.float32)

    body_sigma = N / 6.0
    vol += core_contrast * np.exp(
        -((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * body_sigma ** 2)
    )

    for _ in range(n_blobs):
        r = rng.uniform(N / 10, N / 3.5)
        theta = rng.uniform(0, np.pi)      # polar
        phi = rng.uniform(0, 2 * np.pi)    # azimuth
        bx = c + r * np.sin(theta) * np.cos(phi)
        by = c + r * np.sin(theta) * np.sin(phi)
        bz = c + r * np.cos(theta)
        sigma = rng.uniform(N / 24, N / 10)
        amp = rng.uniform(0.3, 0.9)
        vol += amp * np.exp(
            -((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) / (2 * sigma ** 2)
        )

    if soft_edge:
        r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
        envelope = 0.5 * (1 - np.tanh((r - N * 0.42) / (N * 0.03)))
        vol *= envelope.astype(np.float32)

    vmax = vol.max()
    if vmax > 0:
        vol = vol / vmax
    return vol.astype(np.float32)


# ---------------------------------------------------------------------------
# Random Euler angles.


def random_euler_angles(
    n: int,
    seed: int = 0,
    uniform_sphere: bool = True,
) -> np.ndarray:
    """Return ``(n, 3)`` array of SPIDER ZYZ Euler angles in degrees.

    Columns are ``[phi, theta, psi]`` and match the convention of
    :func:`pyspider.reconstruct.project` (``R = Rz(psi) Ry(theta) Rz(phi)``).

    Parameters
    ----------
    n
        Number of angle triplets to draw.
    seed
        RNG seed.
    uniform_sphere
        If True (default), ``theta`` is drawn so viewing directions are
        uniformly distributed on the sphere (``cos(theta)`` uniform in
        ``[-1, 1]``).  If False, ``theta`` is uniform in ``[0, 180)`` --
        which oversamples the poles and is only occasionally what you want.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 360.0, size=n).astype(np.float32)
    psi = rng.uniform(0.0, 360.0, size=n).astype(np.float32)
    if uniform_sphere:
        theta = np.rad2deg(np.arccos(rng.uniform(-1.0, 1.0, size=n))).astype(np.float32)
    else:
        theta = rng.uniform(0.0, 180.0, size=n).astype(np.float32)
    return np.stack([phi, theta, psi], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2D in-plane rotation + shift helpers.


def _rotate_shift_2d(
    img: np.ndarray,
    angle_deg: float,
    dy: float = 0.0,
    dx: float = 0.0,
) -> np.ndarray:
    """Rotate ``img`` by ``angle_deg`` about its centre, then translate by
    ``(dy, dx)`` pixels.  Bilinear sampling; areas outside the input are 0.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"img must be 2D, got {img.shape}")
    ny, nx = img.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    # Inverse mapping: for each output pixel, find source pixel.
    # Output coord (y - dy, x - dx) relative to centre, then rotate by -angle.
    y_rel = (yy - cy) - dy
    x_rel = (xx - cx) - dx
    src_y = ca * y_rel + sa * x_rel + cy
    src_x = -sa * y_rel + ca * x_rel + cx
    return _bilinear_sample(img, src_y, src_x)


def _bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ny, nx = img.shape
    y0 = np.floor(y).astype(np.int32); y1 = y0 + 1
    x0 = np.floor(x).astype(np.int32); x1 = x0 + 1
    fy = y - y0; fx = x - x0
    inside = (y >= 0) & (x >= 0) & (y <= ny - 1) & (x <= nx - 1)
    y0c = np.clip(y0, 0, ny - 1); y1c = np.clip(y1, 0, ny - 1)
    x0c = np.clip(x0, 0, nx - 1); x1c = np.clip(x1, 0, nx - 1)
    v = ((1 - fy) * ((1 - fx) * img[y0c, x0c] + fx * img[y0c, x1c])
         + fy * ((1 - fx) * img[y1c, x0c] + fx * img[y1c, x1c]))
    return np.where(inside, v, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stack generators.


class RotationStack(NamedTuple):
    """Stack of 2D in-plane rotations + metadata needed to invert them."""
    images: np.ndarray   # (N, H, W) float32
    angles: np.ndarray   # (N,) degrees
    shifts: np.ndarray   # (N, 2) pixels, [dy, dx]


class ProjectionStack(NamedTuple):
    """Stack of 2D projections of a 3D volume + their viewing geometry."""
    images: np.ndarray   # (N, H, W) float32
    eulers: np.ndarray   # (N, 3) degrees, [phi, theta, psi]
    shifts: np.ndarray   # (N, 2) pixels, [dy, dx]


def random_rotations(
    image: np.ndarray,
    n: int,
    seed: int = 0,
    angle_range: tuple[float, float] = (0.0, 360.0),
    shift_sigma: float = 0.0,
    noise_sigma: float = 0.0,
) -> RotationStack:
    """Random in-plane rotations (and optional shifts/noise) of a 2D image.

    Useful for building ``pyspider.align.rotational_angle`` /
    ``translational_shift`` demos and quick classification tests.

    Parameters
    ----------
    image
        Source 2D image (e.g. anything from :mod:`pyspider.testdata`).
    n
        Number of output copies.
    seed
        RNG seed.
    angle_range
        ``(low, high)`` in degrees; angles are drawn uniformly in this
        half-open range.
    shift_sigma
        Standard deviation of per-image Gaussian shift noise (pixels).
        0 → no translation.
    noise_sigma
        Standard deviation of per-pixel additive Gaussian noise applied
        after rotation.  0 → noise-free.

    Returns
    -------
    RotationStack
        ``images`` has shape ``(n, H, W)``; ``angles`` and ``shifts`` give
        the ground-truth transform applied to each slice.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, got {img.shape}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    lo, hi = angle_range
    if not (hi > lo):
        raise ValueError(f"angle_range must have hi > lo, got {angle_range}")

    rng = np.random.default_rng(seed)
    angles = rng.uniform(lo, hi, size=n).astype(np.float32)
    shifts = (rng.standard_normal((n, 2)) * shift_sigma).astype(np.float32) \
        if shift_sigma > 0 else np.zeros((n, 2), dtype=np.float32)

    H, W = img.shape
    out = np.empty((n, H, W), dtype=np.float32)
    for i in range(n):
        rotated = _rotate_shift_2d(img, float(angles[i]),
                                   float(shifts[i, 0]), float(shifts[i, 1]))
        if noise_sigma > 0:
            rotated = rotated + noise_sigma * rng.standard_normal((H, W)).astype(np.float32)
        out[i] = rotated
    return RotationStack(images=out, angles=angles, shifts=shifts)


def random_projections(
    volume: np.ndarray,
    n: int,
    seed: int = 0,
    uniform_sphere: bool = True,
    shift_sigma: float = 0.0,
    noise_sigma: float = 0.0,
) -> ProjectionStack:
    """Stack of random 2D projections of a 3D volume.

    Draws ``n`` random SPIDER ZYZ Euler triplets, calls
    :func:`pyspider.reconstruct.project` for each, and optionally adds
    random in-plane shifts and Gaussian noise.  The returned ``eulers``
    and ``shifts`` are the ground-truth parameters you'd feed back to
    :func:`pyspider.reconstruct.bp3f` (after inverting the shifts) in a
    reconstruction self-consistency test.

    Parameters
    ----------
    volume
        3D NumPy array of shape ``(NZ, NY, NX)``.
    n
        Number of projections to generate.
    seed
        RNG seed.
    uniform_sphere
        See :func:`random_euler_angles`.
    shift_sigma
        Standard deviation of per-image Gaussian shift (pixels) applied
        after projection.  0 → projections centred.
    noise_sigma
        Standard deviation of per-pixel Gaussian noise added last.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got {vol.shape}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    eulers = random_euler_angles(n, seed=seed, uniform_sphere=uniform_sphere)
    # Draw shifts / noise from a child stream so the Euler sequence doesn't
    # change if the user flips shift_sigma / noise_sigma on and off.
    rng = np.random.default_rng(seed + 1)
    shifts = (rng.standard_normal((n, 2)) * shift_sigma).astype(np.float32) \
        if shift_sigma > 0 else np.zeros((n, 2), dtype=np.float32)

    _, ny, nx = vol.shape
    out = np.empty((n, ny, nx), dtype=np.float32)
    for i in range(n):
        proj = reconstruct.project(vol, float(eulers[i, 0]),
                                   float(eulers[i, 1]), float(eulers[i, 2]))
        if shift_sigma > 0:
            proj = _rotate_shift_2d(proj, 0.0,
                                    float(shifts[i, 0]), float(shifts[i, 1]))
        if noise_sigma > 0:
            proj = proj + noise_sigma * rng.standard_normal((ny, nx)).astype(np.float32)
        out[i] = proj.astype(np.float32)
    return ProjectionStack(images=out, eulers=eulers, shifts=shifts)
