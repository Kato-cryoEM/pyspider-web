"""SPIDER FQ filters reimplemented in NumPy.

These functions reproduce the radial-frequency filters that SPIDER's
``FQ`` operation applies inside the Fourier transform (see
``src/fq_q.f`` and ``src/ffilts.f`` in the source tree).  The intent is
that you never need to invoke SPIDER's binary just to filter an image:
``pyspider.filters.filter_image`` is a self-contained NumPy
implementation.

The SPIDER binary remains useful as a *numerical oracle* — the tests
under ``tests/test_filters.py`` show that the agreement with SPIDER is
at the single-precision ULP level.

Coverage:

  ``tophat-lp / tophat-hp``    SPIDER IOPT 1 / 2
  ``gauss-lp  / gauss-hp``     SPIDER IOPT 3 / 4
  ``fermi-lp  / fermi-hp``     SPIDER IOPT 5 / 6
  ``butter-lp / butter-hp``    SPIDER IOPT 7 / 8
  ``raised-cos-lp / raised-cos-hp``  SPIDER IOPT 9 / 10  (from src/fq_q.f)

Each filter operates on a 2D real-space image and returns a 2D
real-space image.  Padding policy matches SPIDER's ``FQ``: by default
the image is doubled in each direction and the new area is filled with
the perimeter mean of the input (see ``src/fq_q.f:79-98``).  Set
``padding='none'`` to reproduce ``FQ NP``.

Conventions:

  - All frequency parameters (``fc``, ``fp``, ``fs``, ``temp``) are in
    cycles per pixel in the working (post-padding) frequency grid, so a
    cutoff of 0.1 means "10% of the way to Nyquist".  This matches
    SPIDER's interpretation of ``PARM1`` / ``PARM2`` after the
    ``PARM1 < 0`` / ``PARM1 > 1`` rescaling in ``src/fq_q.f:128-130``.
  - Anisotropic filters: pass ``fc_y`` (or ``temp_y``, ``fp_y``,
    ``fs_y``) to use a different cutoff in Y than in X.  Default is
    isotropic.
  - High-pass filters preserve the DC term, matching SPIDER's
    ``B(1,1) = AVG`` restore in ``src/fq_q.f:336-337``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# helpers


def _border_mean(img: np.ndarray) -> float:
    """Mean of the pixels around the outermost 1-pixel border.

    Matches src/fq_q.f:81-83 (``AVE = SUM of perimeter / (2*(NX+NY)-4)``).
    """
    ny, nx = img.shape
    s = (img[0, :].sum() + img[-1, :].sum()
         + img[1:-1, 0].sum() + img[1:-1, -1].sum())
    return float(s) / float(2 * (nx + ny) - 4)


def _pad_image(img: np.ndarray, mode: str) -> np.ndarray:
    """Pad img to (2*NY, 2*NX) using the named SPIDER strategy."""
    if mode == "none":
        return np.ascontiguousarray(img, dtype=np.float32)
    ny, nx = img.shape
    if mode == "border":
        fill = _border_mean(img)
    elif mode == "zero":
        fill = 0.0
    else:
        raise ValueError(f"unknown padding mode: {mode!r}")
    pad = np.full((2 * ny, 2 * nx), fill, dtype=np.float32)
    pad[:ny, :nx] = img
    return pad


def _freq_grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """rfft2-shape frequency grid in cycles per pixel."""
    fx = np.fft.rfftfreq(nx).astype(np.float32)[None, :]   # 0 .. 0.5
    fy = np.fft.fftfreq(ny).astype(np.float32)[:, None]    # 0,...,1/2,-1/2,...
    return fx, fy


# ---------------------------------------------------------------------------
# individual filter shapes — return H of shape (ny, nx/2+1), dtype float32


def _isotropic_elliptic_radius(
    fx: np.ndarray, fy: np.ndarray, fc: float, fc_y: float
) -> np.ndarray:
    """Distance metric ``(fx/fc)^2 + (fy/fc_y)^2`` used by SPIDER's
    top-hat and Gaussian filters (src/fq_q.f IOPT 1-4)."""
    return (fx * fx) / (fc * fc) + (fy * fy) / (fc_y * fc_y)


def _radial(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Isotropic radial frequency sqrt(fx² + fy²)."""
    return np.sqrt(fx * fx + fy * fy).astype(np.float32)


def tophat_lp(fx, fy, fc: float, fc_y: float | None = None) -> np.ndarray:
    """SPIDER IOPT 1 — sharp low-pass (passes f² < fc²)."""
    fc_y = fc_y if fc_y is not None else fc
    rho = _isotropic_elliptic_radius(fx, fy, fc, fc_y)
    return (rho <= 1.0).astype(np.float32)


def tophat_hp(fx, fy, fc: float, fc_y: float | None = None) -> np.ndarray:
    """SPIDER IOPT 2 — sharp high-pass (DC preserved)."""
    fc_y = fc_y if fc_y is not None else fc
    rho = _isotropic_elliptic_radius(fx, fy, fc, fc_y)
    H = (rho > 1.0).astype(np.float32)
    H[0, 0] = 1.0
    return H


def gauss_lp(fx, fy, fc: float, fc_y: float | None = None) -> np.ndarray:
    """SPIDER IOPT 3 — exp(-(fx² + fy²)/(2 fc²)) with underflow clamp."""
    fc_y = fc_y if fc_y is not None else fc
    F = 0.5 * ((fx * fx) / (fc * fc) + (fy * fy) / (fc_y * fc_y))
    H = np.where(F < 16.0, np.exp(-F), 0.0).astype(np.float32)
    return H


def gauss_hp(fx, fy, fc: float, fc_y: float | None = None) -> np.ndarray:
    """SPIDER IOPT 4 — 1 - exp(-...), DC preserved."""
    H = (1.0 - gauss_lp(fx, fy, fc, fc_y)).astype(np.float32)
    H[0, 0] = 1.0
    return H


def fermi_lp(fx, fy, fc: float, temp: float) -> np.ndarray:
    """SPIDER IOPT 5 — Fermi-Dirac edge centred at radius fc, width temp."""
    r = _radial(fx, fy)
    arg = np.clip((r - fc) / temp, -10.0, 10.0)
    return (1.0 / (1.0 + np.exp(arg))).astype(np.float32)


def fermi_hp(fx, fy, fc: float, temp: float) -> np.ndarray:
    """SPIDER IOPT 6 — Fermi-Dirac high-pass; equivalent to fermi_lp with
    negative temperature.  DC restored.
    """
    H = fermi_lp(fx, fy, fc, -temp)
    H[0, 0] = 1.0
    return H


def _butter_order(fp: float, fs: float) -> tuple[float, float]:
    """Compute the Butterworth order and normalised cutoff parm1.

    Matches src/fq_q.f:113-119 (EPS=0.882, AA=10.624 — the standard SPIDER
    pass/stop ripple tolerances 0.5 dB / -40 dB).
    """
    EPS = 0.882
    AA = 10.624
    if fp <= 0 or fs <= 0 or fp == fs:
        raise ValueError(f"fp={fp} fs={fs} must be positive and distinct")
    ord_ = 2.0 * np.log10(EPS / np.sqrt(AA * AA - 1.0)) / np.log10(fp / fs)
    parm1 = fp / EPS ** (2.0 / ord_)
    return float(ord_), float(parm1)


def butter_lp(fx, fy, fp: float, fs: float) -> np.ndarray:
    """SPIDER IOPT 7 — Butterworth low-pass; fp = passband, fs = stopband."""
    ord_, parm1 = _butter_order(fp, fs)
    r = _radial(fx, fy)
    return ((1.0 / (1.0 + (r / parm1) ** ord_)) ** 0.5).astype(np.float32)


def butter_hp(fx, fy, fp: float, fs: float) -> np.ndarray:
    """SPIDER IOPT 8 — Butterworth high-pass.  DC preserved."""
    H = (1.0 - butter_lp(fx, fy, fp, fs)).astype(np.float32)
    H[0, 0] = 1.0
    return H


def raised_cos_lp(fx, fy, fp: float, fs: float) -> np.ndarray:
    """SPIDER IOPT 9 — raised-cosine low-pass over the [fp, fs] transition."""
    r = _radial(fx, fy)
    t = (r - fp) / (fs - fp)
    H = 0.5 * (np.cos(np.pi * t) + 1.0)
    H = np.where(t < 0, 1.0, H)
    H = np.where(t > 1, 0.0, H)
    return H.astype(np.float32)


def raised_cos_hp(fx, fy, fp: float, fs: float) -> np.ndarray:
    """SPIDER IOPT 10 — raised-cosine high-pass.

    Unlike the other HP filters (tophat-hp / gauss-hp / fermi-hp /
    butter-hp), SPIDER's IOPT 10 does *not* restore the DC term — see
    ``src/fq_q.f:336`` which lists only IOPTs 2, 4, 6, 8.  The DC is
    therefore killed exactly the way an ordinary HP filter would.
    """
    r = _radial(fx, fy)
    t = (r - fp) / (fs - fp)
    H = 0.5 * (-np.cos(np.pi * t) + 1.0)
    H = np.where(t < 0, 0.0, H)
    H = np.where(t > 1, 1.0, H)
    return H.astype(np.float32)


_FILTERS: dict[str, Callable] = {
    "tophat-lp": tophat_lp,
    "tophat-hp": tophat_hp,
    "gauss-lp": gauss_lp,
    "gauss-hp": gauss_hp,
    "fermi-lp": fermi_lp,
    "fermi-hp": fermi_hp,
    "butter-lp": butter_lp,
    "butter-hp": butter_hp,
    "raised-cos-lp": raised_cos_lp,
    "raised-cos-hp": raised_cos_hp,
}


def list_kinds() -> list[str]:
    """Return the names of all supported filter kinds."""
    return list(_FILTERS)


def filter_response(
    kind: str, ny: int, nx: int, **kwargs
) -> np.ndarray:
    """Return the filter H(fx, fy) on the rfft2 grid of shape (ny, nx/2+1)."""
    if kind not in _FILTERS:
        raise ValueError(
            f"unknown filter kind: {kind!r}. Known: {sorted(_FILTERS)}"
        )
    fx, fy = _freq_grid(ny, nx)
    return _FILTERS[kind](fx, fy, **kwargs)


def filter_image(
    img: np.ndarray,
    kind: str,
    *,
    padding: str = "border",
    **kwargs,
) -> np.ndarray:
    """Apply a SPIDER-style frequency filter to a 2D image.

    Parameters
    ----------
    img
        Input image, shape ``(NY, NX)``.  Anything castable to float32.
    kind
        One of the names in :func:`list_kinds`.
    padding
        ``"border"`` (default; SPIDER ``FQ``), ``"none"`` (SPIDER
        ``FQ NP``), or ``"zero"``.
    **kwargs
        Filter parameters.  See the per-filter functions for the
        accepted argument names (``fc``, ``fc_y``, ``temp``, ``fp``,
        ``fs``).

    Returns
    -------
    np.ndarray
        Filtered image, same shape and float32 dtype as the input.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"img must be 2D, got shape {img.shape}")
    ny, nx = img.shape

    work = _pad_image(img, padding)
    wny, wnx = work.shape

    H = filter_response(kind, wny, wnx, **kwargs)
    spec = np.fft.rfft2(work)
    out = np.fft.irfft2(spec * H, s=work.shape).astype(np.float32)
    return out[:ny, :nx]
