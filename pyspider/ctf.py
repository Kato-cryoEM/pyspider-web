"""Contrast Transfer Function (CTF) for cryo-EM.

This module mirrors SPIDER's CTF math from ``src/tfd.f`` / ``src/trafc.f``:

  - ``ctf_2d``         compute the CTF on a 2D rfft2 grid (matches ``TF C``)
  - ``phase_flip``     correct an image by flipping signs where CTF<0
                       (matches the output of ``TF CT``: ``sign(CTF)``)
  - ``wiener_correct`` simple Wiener filter inverse: ``H / (|H|^2 + 1/SNR)``
  - ``apply_ctf``      multiply a clean image by the CTF in Fourier space
                       (the forward / image-formation model)

Conventions
-----------
- All spatial frequencies are in cycles per Angstrom (1/A).
- Wavelength ``lambda`` is in Angstroms; computed from beam voltage in kV
  by the relativistic formula.
- ``defocus`` (and ``astigmatism``) are in Angstroms (positive = under-focus,
  the convention used by ``tfd.f``).
- ``cs`` (spherical aberration) is in millimetres, matching SPIDER's
  ``TF C`` prompt; internally converted to Angstroms via ``cs * 1e7``.
- ``amplitude_contrast`` ``alpha`` is a dimensionless ratio in [0, 1].
- ``pixel_size`` ``p`` is in Angstroms; the Nyquist frequency is
  ``1 / (2 p)``.

The phase aberration is

    gamma(k) = (pi / 2) * Cs * lambda^3 * k^4  -  pi * dZ * lambda * k^2

and the (envelope-free) CTF is

    H(k) = sin( gamma(k) - W ),     W = arctan(alpha / (1 - alpha))

which matches the literal Fortran of ``tfd.f`` (``QQT`` is gamma in
SPIDER's normalised internal variables).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Physics helpers.


def electron_wavelength(voltage_kV: float) -> float:
    """Relativistic electron wavelength in Angstroms for ``voltage_kV`` kV.

    Standard formula: lambda = h / sqrt(2 m_e e V (1 + e V / (2 m_e c^2))).
    The closed form below evaluates to within ~1e-6 Å of CODATA values.
    """
    V = voltage_kV * 1000.0
    return 12.2639 / np.sqrt(V * (1.0 + 0.97845e-6 * V))


# ---------------------------------------------------------------------------
# Core CTF computation.


def _gamma(k2: np.ndarray, defocus: float, lambda_A: float,
           cs_A: float) -> np.ndarray:
    """Phase aberration gamma(k) on a precomputed |k|^2 grid (k in 1/A)."""
    return 0.5 * np.pi * cs_A * lambda_A ** 3 * k2 * k2 - np.pi * defocus * lambda_A * k2


def _wgh_phase(alpha: float) -> float:
    """SPIDER's WGH = atan(alpha / (1 - alpha)) for amplitude contrast."""
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"amplitude_contrast must be in [0, 1), got {alpha}")
    return float(np.arctan(alpha / (1.0 - alpha))) if alpha > 0 else 0.0


def _rfft2_k_grid(shape: tuple[int, int],
                  pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequency grids (kx, ky) on the rfft2 layout, in 1/Angstrom."""
    ny, nx = shape
    kx = np.fft.rfftfreq(nx, d=pixel_size).astype(np.float32)[None, :]
    ky = np.fft.fftfreq(ny, d=pixel_size).astype(np.float32)[:, None]
    return kx, ky


def ctf_2d(
    shape: tuple[int, int],
    *,
    pixel_size: float,
    defocus: float,
    voltage_kV: float = 200.0,
    cs_mm: float = 2.0,
    amplitude_contrast: float = 0.07,
    astigmatism: float = 0.0,
    azimuth_deg: float = 0.0,
    b_factor: float = 0.0,
) -> np.ndarray:
    """Compute the SPIDER-convention CTF on the rfft2 grid.

    Returns a real array of shape ``(ny, nx // 2 + 1)`` -- the CTF is
    a real function of frequency.  Sign convention matches ``TF C``:

        H(k) = E(k) * sin( gamma(k) - W )

    ``E(k)`` here is a B-factor (Gaussian) envelope only:
    ``E(k) = exp(-b_factor * |k|^2 / 4)``.  SPIDER's full ``tfd.f``
    additionally damps by source-size and defocus-spread envelopes; pass
    ``b_factor`` to enable the analogous overall damping.

    Parameters
    ----------
    shape
        Real-space image shape ``(ny, nx)``.
    pixel_size
        Pixel size in Angstroms.
    defocus
        Defocus in Angstroms.  Positive = under-focus.
    voltage_kV
        Electron beam voltage in kV (used to derive ``lambda``).
    cs_mm
        Spherical aberration in millimetres.
    amplitude_contrast
        Dimensionless amplitude-to-phase contrast ratio in [0, 1).
        Typical ice-embedded biological samples use 0.07.
    astigmatism, azimuth_deg
        Astigmatism magnitude in Angstroms and azimuth in degrees.  Set
        ``astigmatism = 0`` for the isotropic case.
    b_factor
        B-factor envelope in A^2.  ``0`` = no envelope damping.
    """
    lambda_A = electron_wavelength(voltage_kV)
    cs_A = cs_mm * 1.0e7
    W = _wgh_phase(amplitude_contrast)

    kx, ky = _rfft2_k_grid(shape, pixel_size)
    if astigmatism == 0.0:
        k2 = kx * kx + ky * ky
        gamma = _gamma(k2, defocus, lambda_A, cs_A)
    else:
        # Direction-dependent defocus -- matches trafc.f:
        # DZZ = DZ + DZA/2 * sin(2*(AZ - AZR))
        # where AZ is the angle of the (kx, ky) vector + pi/2, AZR the user
        # astigmatism azimuth.  The +pi/2 offset comes from SPIDER's
        # historical convention; we transliterate it verbatim.
        azr = np.deg2rad(azimuth_deg)
        az = np.arctan2(ky, np.where(kx == 0, 1e-30, kx)) + 0.5 * np.pi
        dz_eff = defocus + 0.5 * astigmatism * np.sin(2.0 * (az - azr))
        k2 = kx * kx + ky * ky
        gamma = (0.5 * np.pi * cs_A * lambda_A ** 3 * k2 * k2
                 - np.pi * dz_eff * lambda_A * k2)

    H = np.sin(gamma - W).astype(np.float32)

    if b_factor != 0.0:
        # B-factor envelope: exp(-B * k^2 / 4).  The B-factor convention
        # matches RELION / cryoSPARC: B > 0 damps high frequencies.
        env = np.exp(-b_factor * (kx * kx + ky * ky) / 4.0).astype(np.float32)
        H = H * env
    return H


# ---------------------------------------------------------------------------
# Image-formation model and corrections.


def apply_ctf(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Multiply ``img`` by CTF ``H`` in Fourier space.

    ``img`` may be 2D ``(ny, nx)`` or 3D ``(N, ny, nx)`` — a stack of
    frames — in which case the same CTF is applied to every frame via
    NumPy broadcasting. ``H`` must have shape ``(ny, nx // 2 + 1)`` —
    the rfft2 half-spectrum — as returned by :func:`ctf_2d`.
    """
    img = np.asarray(img, dtype=np.float32)
    axes = (-2, -1)
    spec = np.fft.rfft2(img, axes=axes)
    return np.fft.irfft2(spec * H, s=img.shape[-2:], axes=axes).astype(np.float32)


def phase_flip(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Phase-flip correction: multiply by ``sign(H)``.

    This is exactly what SPIDER's ``TF CT`` writes out (a Fourier file
    of +/-1 entries) and what you apply to a CTF-corrupted image to flip
    the inverted CTF lobes back to a consistent sign.  Information at
    CTF zeros is lost (set to zero); use :func:`wiener_correct` to keep
    it.

    ``img`` may be 2D or a 3D stack ``(N, ny, nx)``; the fix broadcasts.
    """
    img = np.asarray(img, dtype=np.float32)
    axes = (-2, -1)
    spec = np.fft.rfft2(img, axes=axes)
    return np.fft.irfft2(spec * np.sign(H), s=img.shape[-2:], axes=axes).astype(np.float32)


def wiener_correct(img: np.ndarray, H: np.ndarray,
                   snr: float | np.ndarray = 1.0) -> np.ndarray:
    """Wiener-filter inverse CTF:  H_corr = H / (|H|^2 + 1/snr).

    For ``snr -> infinity`` this is ``1 / H`` (pure inverse); for
    ``snr -> 0`` it goes to zero.  The default ``snr = 1`` matches the
    classical regularised inverse used as a starting point in many
    refinement pipelines.

    ``img`` may be 2D or a 3D stack ``(N, ny, nx)``.
    """
    img = np.asarray(img, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    snr_inv = np.float32(1.0 / float(snr))
    weight = H / (H * H + snr_inv)
    axes = (-2, -1)
    spec = np.fft.rfft2(img, axes=axes)
    return np.fft.irfft2(spec * weight, s=img.shape[-2:], axes=axes).astype(np.float32)
