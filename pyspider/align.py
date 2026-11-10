"""2D translational and rotational alignment in pure NumPy.

This module mirrors SPIDER's foundational alignment primitives:

  ``cross_correlate``    SPIDER ``CC`` (``corr1.f`` -> ``ccrs.f``):
                         inverse FFT of ``conj(F[a]) * F[b]`` with the
                         peak of zero shift placed at the centre of the
                         output map (an implicit ``fftshift``).
  ``find_peaks``         SPIDER ``PK`` (``speak.f``): scan for strict
                         local maxima of a 2D map and report the top
                         ``n_peaks`` with sub-pixel refinement.
  ``parabolic_refine``   SPIDER ``PARABL`` (``parabl.f``): fit a
                         paraboloid to a 3x3 neighbourhood and return
                         the analytic peak shift in ``[-1, 1]``.
  ``translational_shift``  Convenience wrapper that ties the above
                         together: align ``b`` to ``a`` and return the
                         sub-pixel ``(dy, dx)`` offset.

  ``polar_sample``       Bilinear interpolation from a Cartesian image
                         onto a polar (radius x angle) grid.  Same
                         underlying math as SPIDER's ring sampler
                         (``aprings_*.f`` / ``apsh_*.f``), simplified
                         to the pedagogical core.
  ``rotational_correlate`` 1D angular cross-correlation of the polar
                         transforms of two images, summed across rings.
                         The 1D analogue of ``cross_correlate``.
  ``rotational_angle``   Returns the sub-pixel angle (degrees) that
                         rotates ``b`` back into alignment with ``a``.

All functions stay strictly in NumPy / ``numpy.fft``; the SPIDER binary
is only used as the validation oracle in ``tests/test_align.py``.

Coordinate conventions
----------------------
- Image arrays are indexed ``img[y, x]`` (NumPy default), so we always
  return shifts as ``(dy, dx)``.
- ``cross_correlate`` returns a real array the same shape as the inputs.
  The pixel ``cc[ny//2, nx//2]`` corresponds to **zero shift** (this
  matches SPIDER's ``CC`` output and lets a NumPy user use
  ``np.argmax`` to read a shift directly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# PARABL (parabl.f) -- analytic peak of a quadratic surface fit to a 3x3
# neighbourhood.  Z[0,0] is the upper-left of the patch, Z[2,2] the
# lower-right, and Z[1,1] the discrete maximum.
#
# Following ``src/parabl.f``, the fitted surface is
#
#   F(x, y) = C1 + C2*y + C3*y**2 + C4*x + C5*xy + C6*x**2
#
# where (x, y) range over {1, 2, 3} in the 3x3 patch and the returned
# (xsh, ysh) are the peak position relative to the *centre* of the
# patch, i.e. zero means "the discrete maximum is already optimal" and
# +/-1 means "the maximum lies on a neighbour".


def parabolic_refine(z: np.ndarray) -> tuple[float, float, float]:
    """Return (ysh, xsh, peak_value) for a 3x3 patch ``z``.

    Input is assumed to be in NumPy convention ``z[y, x]`` (row index
    first).  SPIDER's PARABL uses the opposite convention
    ``Z[i, j] = F(x=i, y=j)``, so we transpose internally before
    applying the closed-form coefficients from ``src/parabl.f``.

    The shifts are clamped to ``[-1, 1]`` to match PARABL.
    """
    if z.shape != (3, 3):
        raise ValueError(f"parabolic_refine expects a 3x3 patch, got {z.shape}")
    # PARABL convention is Z(x, y); NumPy gives us z[y, x] -> transpose.
    Z = z.astype(np.float64).T

    C1 = (26.0*Z[0,0] - Z[0,1] + 2.0*Z[0,2]
          - Z[1,0] - 19.0*Z[1,1] - 7.0*Z[1,2]
          + 2.0*Z[2,0] - 7.0*Z[2,1] + 14.0*Z[2,2]) / 9.0
    C2 = (8.0*Z[0,0] - 8.0*Z[0,1]
          + 5.0*Z[1,0] - 8.0*Z[1,1] + 3.0*Z[1,2]
          + 2.0*Z[2,0] - 8.0*Z[2,1] + 6.0*Z[2,2]) / -6.0
    C3 = (Z[0,0] - 2.0*Z[0,1] + Z[0,2]
          + Z[1,0] - 2.0*Z[1,1] + Z[1,2]
          + Z[2,0] - 2.0*Z[2,1] + Z[2,2]) / 6.0
    C4 = (8.0*Z[0,0] + 5.0*Z[0,1] + 2.0*Z[0,2]
          - 8.0*Z[1,0] - 8.0*Z[1,1] - 8.0*Z[1,2]
          + 3.0*Z[2,1] + 6.0*Z[2,2]) / -6.0
    C5 = (Z[0,0] - Z[0,2] - Z[2,0] + Z[2,2]) / 4.0
    C6 = (Z[0,0] + Z[0,1] + Z[0,2]
          - 2.0*Z[1,0] - 2.0*Z[1,1] - 2.0*Z[1,2]
          + Z[2,0] + Z[2,1] + Z[2,2]) / 6.0

    denom = 4.0 * C3 * C6 - C5 * C5
    if denom == 0.0:
        # Degenerate fit -- fall back to the discrete maximum.
        return 0.0, 0.0, float(z[1, 1])

    ysh = (C4 * C5 - 2.0 * C2 * C6) / denom - 2.0
    xsh = (C2 * C5 - 2.0 * C4 * C3) / denom - 2.0
    peak = (4.0 * C1 * C3 * C6 - C1 * C5 * C5
            - C2 * C2 * C6 + C2 * C4 * C5 - C4 * C4 * C3) / denom

    # Clamp the same way ``PARABL`` does (src/parabl.f:75-78).
    ysh = max(-1.0, min(1.0, ysh))
    xsh = max(-1.0, min(1.0, xsh))
    # PARABL's argument ordering puts X (column) first in the call signature
    # but the math derives YSH first; we return (ysh, xsh) to stay aligned
    # with NumPy's row-major convention.
    return float(ysh), float(xsh), float(peak)


# ---------------------------------------------------------------------------
# CC (corr1.f -> ccrs.f) -- inverse FFT of conj(F[a]) * F[b] with an
# implicit fftshift baked into the Fourier-space multiply.


def cross_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-correlation of two 2D images, SPIDER ``CC`` convention.

    Returns a real-valued array the same shape as the inputs.  The pixel
    at index ``(ny // 2, nx // 2)`` is the correlation at zero shift.
    A peak at ``(ny // 2 + dy, nx // 2 + dx)`` means **shifting** ``b``
    by ``(dy, dx)`` aligns it with ``a`` (equivalently, ``a`` already
    holds ``b`` shifted by ``(-dy, -dx)``).

    The math (mirroring ``ccrs.f`` once both ends of the sign-convention
    chain are reconciled):

        cc = ifft( F[a] * conj(F[b]) )
        cc = fftshift(cc)               # zero shift at centre

    Note that SPIDER's FT uses ``exp(+i)`` (opposite of NumPy), which
    flips which operand gets the conjugate when expressed in NumPy's
    ``exp(-i)`` convention — hence ``conj(F[b])`` here even though
    ``ccrs.f`` literally writes ``conj(X) * Y``.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"need 2D inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    return np.fft.fftshift(cc).astype(np.float32)


# ---------------------------------------------------------------------------
# PK (speak.f) -- find local maxima and refine each with PARABL.


@dataclass
class Peak:
    """One peak as reported by :func:`find_peaks`.

    Attributes
    ----------
    iy, ix
        Integer-pixel coordinates of the local maximum (0-based).
    dy, dx
        Sub-pixel refinement from ``parabolic_refine``.  The refined
        peak coordinate is ``(iy + dy, ix + dx)``.
    height
        Value of the discrete maximum (matches SPIDER's ALIST entry).
    refined_height
        Value of the paraboloid at its analytic peak (matches SPIDER's
        ``PEAKV``).  ``None`` for boundary peaks where no 3x3
        neighbourhood is available.
    """

    iy: int
    ix: int
    dy: float
    dx: float
    height: float
    refined_height: float | None


def find_peaks(
    cc: np.ndarray,
    *,
    n_peaks: int = 1,
    sub_pixel: bool = True,
) -> list[Peak]:
    """Find the highest local maxima of ``cc``.

    A pixel is considered a peak when it is **strictly greater** than
    each of its 8 immediate neighbours -- matching SPIDER's strict
    inequality test (``speak.f:141-148``).

    Peaks within one pixel of the boundary are excluded because they
    lack a full 3x3 neighbourhood; this also matches ``speak.f`` which
    iterates ``K = 2, NX-1`` and ``I = 3, NY``.
    """
    if cc.ndim != 2:
        raise ValueError(f"need a 2D map, got {cc.shape}")
    if n_peaks < 1:
        raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")
    a = np.asarray(cc, dtype=np.float32)
    ny, nx = a.shape

    # Vectorised strict-local-max test over the interior.
    centre = a[1:-1, 1:-1]
    neigh = [
        a[ :-2,  :-2], a[ :-2, 1:-1], a[ :-2, 2:],
        a[1:-1,  :-2],                a[1:-1, 2:],
        a[2:  ,  :-2], a[2:  , 1:-1], a[2:  , 2:],
    ]
    is_max = np.ones_like(centre, dtype=bool)
    for n in neigh:
        is_max &= (centre > n)

    if not is_max.any():
        return []

    # Collect (height, iy, ix) for every strict local max.
    iy, ix = np.nonzero(is_max)
    heights = centre[iy, ix]
    # Translate from interior coords back to full-image coords.
    iy += 1
    ix += 1

    # Sort descending by height and keep the top n_peaks.
    order = np.argsort(-heights, kind="stable")[:n_peaks]

    peaks: list[Peak] = []
    for k in order:
        py, px, h = int(iy[k]), int(ix[k]), float(heights[k])
        if sub_pixel:
            patch = a[py - 1 : py + 2, px - 1 : px + 2]
            dy, dx, refined = parabolic_refine(patch)
        else:
            dy = dx = 0.0
            refined = h
        peaks.append(Peak(iy=py, ix=px, dy=dy, dx=dx,
                          height=h, refined_height=refined))
    return peaks


# ---------------------------------------------------------------------------
# Convenience: full translational alignment.


def translational_shift(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sub_pixel: bool = True,
) -> tuple[float, float, float]:
    """Estimate the integer + sub-pixel shift that aligns ``b`` to ``a``.

    Returns ``(dy, dx, peak)`` where ``(dy, dx)`` is the offset to apply
    to ``b`` to register it onto ``a``: that is, ``a`` matches
    ``b`` shifted by ``(-dy, -dx)``.  ``peak`` is the refined
    correlation value at the peak.

    Conventions match SPIDER ``CC`` + ``PK`` :
      - centre of the cross-correlation map is zero shift;
      - peak coordinates are reported with sub-pixel refinement.
    """
    cc = cross_correlate(a, b)
    peaks = find_peaks(cc, n_peaks=1, sub_pixel=sub_pixel)
    if not peaks:
        raise RuntimeError("no peaks found in cross-correlation")
    p = peaks[0]
    ny, nx = a.shape
    # The peak at exactly the centre corresponds to zero shift.
    dy = float(p.iy + p.dy - ny // 2)
    dx = float(p.ix + p.dx - nx // 2)
    return dy, dx, float(p.refined_height) if p.refined_height is not None else float(p.height)


# ---------------------------------------------------------------------------
# Polar resampling and rotational alignment.
#
# The classical SPIDER recipe (aprings_*.f / apsh_*.f, with all the FFTW
# plan caching and MPI machinery stripped away):
#
#   1. Sample each input image on a polar grid (concentric rings of
#      radii r_1, ..., r_R, sampled at A equally-spaced angles).
#   2. For each ring, compute the 1D cross-correlation between the two
#      images along the angle axis.  This is the standard "rotation is
#      translation in polar" trick.
#   3. Sum the per-ring 1D CCs (optionally weighted) into a single
#      "rotational correlation function" defined over the angle axis.
#   4. The location of its maximum, refined by 1D parabolic fitting, is
#      the relative rotation angle.
#
# This module keeps the simplest version of each step.  SPIDER's own
# implementation additionally rounds each ring's length to a power of
# two to use a cached FFTW plan; uses ring weights and Joyeux-style
# normalisation; and bundles in-place mirror checks.  Those are pure
# performance / robustness optimisations and don't change the underlying
# math.


def polar_sample(
    img: np.ndarray,
    *,
    n_rings: int,
    n_angles: int,
    radii: np.ndarray | None = None,
    centre: tuple[float, float] | None = None,
) -> np.ndarray:
    """Resample a 2D image onto a polar (radius x angle) grid.

    Returns an array of shape ``(n_rings, n_angles)`` where row ``i``
    holds the image values along the circle of radius ``radii[i]`` (or,
    if ``radii`` is None, ``radii = linspace(1, r_max, n_rings)`` with
    ``r_max = min(ny, nx) / 2 - 1``).  Angles run from 0 (positive x)
    counter-clockwise.

    Implementation: bilinear interpolation, matching the linear
    interpolation that ``aprings_*.f`` uses for its ring sampler.
    Out-of-image coordinates return 0.
    """
    img = np.ascontiguousarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"img must be 2D, got shape {img.shape}")
    ny, nx = img.shape
    if centre is None:
        cy, cx = (ny - 1) * 0.5, (nx - 1) * 0.5
    else:
        cy, cx = float(centre[0]), float(centre[1])

    if radii is None:
        r_max = float(min(ny, nx) / 2 - 1)
        radii = np.linspace(1.0, r_max, n_rings, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(n_rings, 1)

    # Angles: equally spaced over [0, 2pi).
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float64)
    cos_t = np.cos(theta).reshape(1, n_angles)
    sin_t = np.sin(theta).reshape(1, n_angles)

    # Cartesian sample coordinates.
    xs = cx + radii * cos_t   # shape (n_rings, n_angles)
    ys = cy + radii * sin_t

    # Bilinear interpolation.
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = xs - x0
    fy = ys - y0

    # In-bounds mask; out-of-image samples become 0.
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < nx) & (y1 < ny)
    x0 = np.clip(x0, 0, nx - 1)
    x1 = np.clip(x1, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1)
    y1 = np.clip(y1, 0, ny - 1)

    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]
    val = ((1 - fx) * (1 - fy) * v00
           + fx * (1 - fy) * v01
           + (1 - fx) * fy * v10
           + fx * fy * v11)
    val = np.where(inside, val, 0.0)
    return val.astype(np.float32)


def rotational_correlate(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_rings: int = 32,
    n_angles: int = 360,
    radii: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    centre: tuple[float, float] | None = None,
    normalize_rings: bool = True,
) -> np.ndarray:
    """1D rotational cross-correlation, summed over rings.

    Returns a 1D real array of length ``n_angles``.  Element ``k``
    measures how well ``b`` matches ``a`` when rotated by
    ``360 * k / n_angles`` degrees.  Peak at ``k`` <=> rotating
    ``b`` by ``(360 * k / n_angles) deg`` aligns it with ``a``.

    ``normalize_rings`` (default True) subtracts the mean and divides
    by the std of each polar ring **independently** before correlating,
    turning the per-ring sum into a sum of Pearson coefficients. This
    matches SPIDER AP MR's "normalised cross-correlation" behaviour.
    Without it, high-variance references (e.g. equator views of a
    tetrameric particle) monopolise the match score simply because
    their rings carry more total energy than oblique views.

    The sum is optionally weighted per ring (``weights`` of length
    ``n_rings``); SPIDER's default is ``weights[i] = radii[i]`` which
    gives equal area weight to each ring.  Default here is uniform.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    pa = polar_sample(a, n_rings=n_rings, n_angles=n_angles,
                      radii=radii, centre=centre)
    pb = polar_sample(b, n_rings=n_rings, n_angles=n_angles,
                      radii=radii, centre=centre)

    if normalize_rings:
        # Per-ring zero-mean + unit-std. np.where guards rings that
        # happen to be all-zero (e.g. radius larger than particle extent).
        for arr in (pa, pb):
            m = arr.mean(axis=1, keepdims=True)
            s = arr.std(axis=1, keepdims=True)
            arr -= m
            np.divide(arr, s, out=arr, where=s > 0)

    # 1D angular cross-correlation per ring; same sign convention as
    # ``cross_correlate``: CC peak at offset k means rotating b by k
    # angle-bins aligns it with a.
    fa = np.fft.rfft(pa, axis=1)
    fb = np.fft.rfft(pb, axis=1)
    cc = np.fft.irfft(fa * np.conj(fb), n=n_angles, axis=1).astype(np.float32)

    if weights is None:
        return cc.sum(axis=0)
    weights = np.asarray(weights, dtype=np.float32).reshape(n_rings, 1)
    return (cc * weights).sum(axis=0)


def _parabolic_refine_1d(y0: float, y1: float, y2: float) -> tuple[float, float]:
    """1D 3-point parabolic refinement.  Returns (sub-pixel offset in
    [-1, 1], refined peak value).  Mirrors PARABL's clamping rule.
    """
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return 0.0, float(y1)
    dx = 0.5 * (y0 - y2) / denom
    dx = max(-1.0, min(1.0, float(dx)))
    peak = float(y1 - 0.25 * (y0 - y2) * dx)
    return dx, peak


def rotational_angle(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_rings: int = 32,
    n_angles: int = 360,
    radii: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    centre: tuple[float, float] | None = None,
    sub_pixel: bool = True,
    normalize_rings: bool = True,
) -> tuple[float, float]:
    """Estimate the rotation angle (degrees) that aligns ``b`` to ``a``.

    Returns ``(angle_deg, peak_value)``.  A positive ``angle_deg`` means
    rotating ``b`` counter-clockwise by ``angle_deg`` degrees brings it
    into best alignment with ``a``.

    ``n_angles`` controls the discrete angular resolution; with sub-pixel
    refinement on, the actual precision is sub-degree even for
    ``n_angles = 360``.
    """
    cc = rotational_correlate(a, b,
                              n_rings=n_rings, n_angles=n_angles,
                              radii=radii, weights=weights, centre=centre,
                              normalize_rings=normalize_rings)
    k = int(np.argmax(cc))
    if not sub_pixel:
        return 360.0 * k / n_angles, float(cc[k])

    # Wrap-around neighbours for sub-pixel parabolic fit.
    km1 = (k - 1) % n_angles
    kp1 = (k + 1) % n_angles
    dx, peak = _parabolic_refine_1d(float(cc[km1]), float(cc[k]), float(cc[kp1]))
    angle = 360.0 * (k + dx) / n_angles
    return angle, peak
