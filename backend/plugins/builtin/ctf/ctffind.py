"""Built-in node — ctf.ctffind (CTFFIND-style defocus estimation).

Estimates the defocus (and optionally astigmatism) of a real-space image
by matching a theoretical CTF² to the observed radial power spectrum,
following the algorithm popularised by Rohou & Grigorieff (CTFFIND4).
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _fn_ctffind


register(
    N("ctf.ctffind", "CTFFIND · Defocus estimate", "CTF",
      "image:image",
      "defocus_u_A:scalar,defocus_v_A:scalar,azimuth_deg:scalar,"
      "score:scalar,theoretical_ctf2:image",
      [
          ("voltage_kV",   "float", 300.0, None, 80.0,   1000.0),
          ("cs_mm",        "float", 2.7,   None, 0.0,     10.0),
          ("amp_contrast", "float", 0.1,   None, 0.0,      1.0),
          ("pixel_A",      "float", 1.0,   None, 0.1,     10.0),
          ("defocus_min_A", "float", 5000.0,  None, 100.0,  100000.0),
          ("defocus_max_A", "float", 50000.0, None, 100.0,  100000.0),
          ("defocus_step_A","float", 500.0,   None, 10.0,    5000.0),
          ("ring_min_frac", "float", 0.05, None, 0.0, 1.0),
          ("ring_max_frac", "float", 0.4,  None, 0.0, 1.0),
          ("fit_astigmatism", "select", "no", ["no", "yes"]),
          ("tile_size",   "int",   512, None, 0, 4096),
          ("tile_overlap","float", 0.5, None, 0.0, 0.95),
      ],
      "Estimate CTF defocus by matching a theoretical CTF² to the observed "
      "power spectrum. Feed a micrograph (or any image with visible Thon "
      "rings) in; the node **tiles the image into small overlapping windows "
      "(default 512² @ 50% overlap), Hann-windows each, and averages the "
      "|FFT|² over all tiles** (CTFFIND4-style preprocessing) to suppress "
      "shot noise before radially averaging. Then it searches defocus "
      "values in [defocus_min_A, defocus_max_A] with defocus_step_A "
      "resolution. The fit uses the spatial-frequency ring [ring_min_frac, "
      "ring_max_frac] × Nyquist. Set fit_astigmatism='yes' for a coarse "
      "(Δf_u, Δf_v, azimuth) grid refinement (slower). Set tile_size=0 to "
      "disable tiling and use one whole-image FFT (e.g. for tests on small "
      "synthetic data). Outputs: defocus_u_A / defocus_v_A (Å, isotropic "
      "mode returns defocus_u=defocus_v), azimuth_deg, score (normalised "
      "cross-correlation, 0..1), and theoretical_ctf2 (2D image, same "
      "size as the tile, for visual comparison with the observed power "
      "spectrum). Standard microscope: 300 kV / Cs 2.7 mm / amp_contrast "
      "0.07–0.10 / pixel_A 1.0–2.0.",
      _fn_ctffind),
)
