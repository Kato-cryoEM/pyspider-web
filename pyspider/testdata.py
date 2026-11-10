"""Test images for the textbook notebooks.

Two flavours:
  - synthetic patterns that demonstrate specific image-processing
    phenomena (Siemens star for resolution, checkerboard for aliasing)
  - real-world photos from ``skimage.data`` (the camera and astronaut
    images replace the historical Lena image, which is no longer shipped
    by mainstream packages).

All helpers return ``float32`` arrays normalised to ``[0, 1]`` (unless
``normalize=False``).  When a target size is given the image is
centre-cropped and resized.
"""

from __future__ import annotations

import numpy as np


def _to_float01(arr: np.ndarray, normalize: bool = True) -> np.ndarray:
    arr = arr.astype(np.float32)
    if normalize:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    return arr


def _square_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop to square, then bilinear resize to (size, size)."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    sq = img[y0:y0 + s, x0:x0 + s]
    if s == size:
        return sq.astype(np.float32)
    # Bilinear resize via index mapping
    ys = np.linspace(0, s - 1, size, dtype=np.float32)
    xs = np.linspace(0, s - 1, size, dtype=np.float32)
    YS, XS = np.meshgrid(ys, xs, indexing="ij")
    y0i = np.floor(YS).astype(np.int32); y1i = np.clip(y0i + 1, 0, s - 1)
    x0i = np.floor(XS).astype(np.int32); x1i = np.clip(x0i + 1, 0, s - 1)
    fy = YS - y0i; fx = XS - x0i
    out = ((1 - fy) * ((1 - fx) * sq[y0i, x0i] + fx * sq[y0i, x1i])
           + fy * ((1 - fx) * sq[y1i, x0i] + fx * sq[y1i, x1i]))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic test patterns.


def siemens_star(N: int = 256, n_spokes: int = 24,
                 smooth: bool = False) -> np.ndarray:
    """Classic Siemens star: alternating wedges in a circular mask.

    The angular frequency of the pattern increases toward the centre,
    so a low-pass filter that smears below some radius is instantly
    visible.

    Parameters
    ----------
    N
        Output size (NxN).
    n_spokes
        Number of black/white wedge pairs.  Higher = finer detail near
        the centre.
    smooth
        If True, use a sinusoidal pattern (smooth contrast); if False
        (default), hard black/white wedges.
    """
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    theta = np.arctan2(y - cy, x - cx)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    R_max = N / 2 * 0.92
    if smooth:
        pattern = 0.5 + 0.5 * np.cos(n_spokes * theta)
    else:
        pattern = (np.cos(n_spokes * theta) > 0).astype(np.float32)
    mask = (r < R_max).astype(np.float32)
    return (pattern * mask).astype(np.float32)


def checkerboard(N: int = 256, square_size: int = 16) -> np.ndarray:
    """Checker pattern -- exposes aliasing, frequency content."""
    y, x = np.mgrid[0:N, 0:N]
    pat = (((x // square_size) + (y // square_size)) % 2).astype(np.float32)
    return pat


def bottle_cap(N: int = 256, n_teeth: int = 36,
               mark_angle_deg: float = 45.0) -> np.ndarray:
    """A bottle-cap / coin-style pattern.

    Built from circularly-symmetric features:
      - an outer filled disk (the "metal of the cap")
      - three concentric bright rings
      - cog-like teeth around the rim
      - an off-centre asymmetric mark whose angular position is
        controlled by ``mark_angle_deg`` (so two near-identical caps
        differing only in mark position can serve as a "similar but
        distinguishable" pair of classes).
    """
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    R_outer = N / 2 * 0.92

    img = np.zeros((N, N), dtype=np.float32)
    # outer disk (cap body, mid grey)
    img += (r < R_outer).astype(np.float32) * 0.45
    # three concentric bright rings
    for r_target, width in [(R_outer * 0.70, 3.0),
                            (R_outer * 0.50, 3.0),
                            (R_outer * 0.30, 2.5)]:
        img += 0.35 * np.exp(-((r - r_target) ** 2) / (2 * width ** 2))
    # cog teeth around the rim
    teeth_band = ((r > R_outer * 0.80) & (r < R_outer * 0.92)).astype(np.float32)
    img += teeth_band * 0.5 * (0.5 + 0.5 * np.cos(n_teeth * theta))
    # asymmetric mark at user-specified angle, mid radius
    r_mark = R_outer * 0.55
    a_mark = np.deg2rad(mark_angle_deg)
    bx = cx + r_mark * np.cos(a_mark)
    by = cy + r_mark * np.sin(a_mark)
    img += 0.9 * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * 7.0 ** 2))
    return _to_float01(img)


def face(N: int = 128, eyes: str = "open") -> np.ndarray:
    """Cartoonish face: outline + eyes + nose + mouth.

    Two faces with ``eyes='open'`` vs ``eyes='closed'`` are visually
    similar (same outline, nose, mouth, same skin tone) but distinguishable
    by the eye region only.  Useful as a "similar but different" pair for
    classification demos where rotation is *not* a confound.

    Parameters
    ----------
    N
        Output size.
    eyes
        ``"open"`` (pupils visible) or ``"closed"`` (horizontal slits).
    """
    if eyes not in ("open", "closed"):
        raise ValueError(f"eyes must be 'open' or 'closed', got {eyes!r}")
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    img = np.zeros((N, N), dtype=np.float32)

    # Face outline (filled disk) with soft edge.
    R = N * 0.36
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    face_mask = 0.5 * (1 - np.tanh((r - R) / (N * 0.012)))
    img += face_mask * 0.60   # skin tone

    # Eye positions
    eye_dx = N * 0.13
    eye_dy = -N * 0.08
    left_x = cx - eye_dx
    right_x = cx + eye_dx
    eye_y = cy + eye_dy
    eye_sx = N * 0.045
    eye_sy = N * 0.030

    if eyes == "open":
        # Sockets (lighter) + pupils (darker)
        for ex in (left_x, right_x):
            socket = np.exp(-((x - ex) ** 2 / (2 * eye_sx ** 2)
                              + (y - eye_y) ** 2 / (2 * eye_sy ** 2)))
            img += socket * 0.35
        for ex in (left_x, right_x):
            pupil = np.exp(-((x - ex) ** 2 + (y - eye_y) ** 2)
                           / (2 * (N * 0.022) ** 2))
            img -= pupil * 0.65
    else:  # closed
        # Horizontal slits across each eye position
        for ex in (left_x, right_x):
            slit = np.exp(-((y - eye_y) ** 2) / (2 * (N * 0.006) ** 2))
            slit *= np.exp(-((x - ex) ** 2) / (2 * eye_sx ** 2))
            img -= slit * 0.55

    # Nose
    nose_y = cy + N * 0.04
    img += 0.12 * np.exp(-((x - cx) ** 2 + (y - nose_y) ** 2)
                         / (2 * (N * 0.022) ** 2))

    # Mouth (smiling arc)
    mouth_cx, mouth_cy = cx, cy + N * 0.20
    mouth_rx, mouth_ry = N * 0.10, N * 0.05
    # ellipse arc, just the lower half
    ellipse = ((x - mouth_cx) ** 2 / mouth_rx ** 2
               + (y - mouth_cy) ** 2 / mouth_ry ** 2)
    mouth_band = np.exp(-((ellipse - 1) ** 2) / (2 * 0.18 ** 2))
    mouth_band *= (y > mouth_cy).astype(np.float32)
    img -= mouth_band * 0.4

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def resolution_target(N: int = 256) -> np.ndarray:
    """USAF-style 3-bar resolution target laid out across the image.

    A row of decreasing bar pairs -- the eye can read off the smallest
    bar pair that survives a filter.
    """
    img = np.zeros((N, N), dtype=np.float32)
    # 5 groups of decreasing bar width
    widths = [12, 8, 5, 3, 2]
    y0 = N // 2 - 30
    y1 = N // 2 + 30
    x = 10
    for w in widths:
        for i in range(3):
            img[y0:y1, x:x + w] = 1.0
            x += 2 * w
        x += 20
    return img


# ---------------------------------------------------------------------------
# Real photographs from skimage.data.


def camera(size: int = 256) -> np.ndarray:
    """Grayscale photo of a man with a tripod camera (skimage.data.camera).

    Replaces the historical Lena image as a classic image-processing
    test photo.  Resized to ``(size, size)``.
    """
    try:
        from skimage import data
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for pyspider.testdata.camera(); "
            "install with 'pip install scikit-image'"
        ) from e
    raw = data.camera()
    return _to_float01(_square_crop_resize(raw, size))


def astronaut(size: int = 256, gray: bool = True) -> np.ndarray:
    """Eileen Collins, astronaut.  Color (gray=False) or B&W (default)."""
    try:
        from skimage import data
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for pyspider.testdata.astronaut()"
        ) from e
    raw = data.astronaut()
    if gray:
        # NTSC luminance.
        raw = (0.299 * raw[..., 0] + 0.587 * raw[..., 1]
               + 0.114 * raw[..., 2])
    return _to_float01(_square_crop_resize(raw, size))


def coins(size: int = 256) -> np.ndarray:
    """Photo of metal coins -- many sharp edges & circular features."""
    try:
        from skimage import data
    except ImportError as e:
        raise ImportError("scikit-image is required") from e
    raw = data.coins()
    return _to_float01(_square_crop_resize(raw, size))


def cell(size: int = 256) -> np.ndarray:
    """Microscopy image of a cell -- biology-flavoured test pattern."""
    try:
        from skimage import data
    except ImportError as e:
        raise ImportError("scikit-image is required") from e
    raw = data.cell()
    return _to_float01(_square_crop_resize(raw, size))


# ---------------------------------------------------------------------------
# Multi-feature synthetic particle (no external dependency).


def cryo_em_particle(N: int = 128, seed: int = 0,
                     bg_level: float = 0.55,
                     noise: float = 0.02) -> np.ndarray:
    """Cryo-EM style particle: dark blob structure on a uniform medium-grey
    "vitreous ice" background.

    Unlike :func:`synthetic_particle` (background = 0 after normalisation),
    this image keeps a clear ``bg_level`` so SPIDER operations like
    padding (`PD`) and windowing (`WI`) show obvious contrast between the
    image area and the padded / cropped region.

    The particle is built from a soft central body plus 6 internal
    sub-domains; protein attenuates the beam so it appears *darker* than
    the surrounding ice (standard cryo-EM convention).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    cx, cy = (N - 1) / 2, (N - 1) / 2

    img = np.full((N, N), bg_level, dtype=np.float32)

    body_r = N / 5
    body = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * body_r ** 2))
    img -= body * 0.32

    for _ in range(6):
        r = rng.uniform(N / 12, N / 6)
        theta = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(N / 18, N / 9)
        bx = cx + r * np.cos(theta)
        by = cy + r * np.sin(theta)
        img -= 0.18 * np.exp(-((x - bx) ** 2 + (y - by) ** 2)
                              / (2 * radius ** 2))

    img += noise * rng.standard_normal((N, N)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def synthetic_particle(N: int = 128, seed: int = 0) -> np.ndarray:
    """A synthetic 'particle' image: 6-8 overlapping Gaussian blobs that
    together suggest a biological structure -- useful when you want a
    realistic look without depending on skimage."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    img = np.zeros((N, N), dtype=np.float32)
    # Main body
    img += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (N / 6) ** 2)) * 0.7
    # Several offset blobs
    n_blobs = 7
    for _ in range(n_blobs):
        r = rng.uniform(N / 8, N / 3)
        theta = rng.uniform(0, 2 * np.pi)
        bx = cx + r * np.cos(theta); by = cy + r * np.sin(theta)
        sigma = rng.uniform(N / 30, N / 12)
        amp = rng.uniform(0.3, 1.0)
        img += amp * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * sigma ** 2))
    return _to_float01(img)
