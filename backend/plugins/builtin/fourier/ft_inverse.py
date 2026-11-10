"""Built-in node — ft.inverse (Inverse FFT, 2D/3D) — Fourier.

Unified 2D / 3D inverse FFT. ``dims`` defaults to ``auto`` and picks the
mode based on which input is wired: ``fourier3d`` → 3D IFT emitting a
volume, otherwise 2D IFT emitting an image.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, Port
from backend.nodes import (_fn_ift)


register(N(
    "ft.inverse", "Inverse FFT", "Fourier",
    [
        Port("fourier",   "fourier",   optional=True,
             depends_on=("dims", ["auto", "2d"])),
        Port("fourier3d", "fourier3d", optional=True,
             depends_on=("dims", ["auto", "3d"])),
    ],
    [
        Port("image",  "image",  depends_on=("dims", ["auto", "2d"])),
        Port("volume", "volume", depends_on=("dims", ["auto", "3d"])),
    ],
    [
        Param("dims", "select", "auto", ["auto", "2d", "3d"]),
        Param("ny", "int", 0, min=0, max=8192,
              depends_on=("dims", ["auto", "2d"])),
        Param("nx", "int", 0, min=0, max=8192,
              depends_on=("dims", ["auto", "2d"])),
    ],
    "Inverse FFT, matching `ft.forward`.\n\n"
    "**`dims=auto`** (default): if `fourier3d` is wired, do a 3D IFT and "
    "emit a volume; otherwise do a 2D IFT from `fourier` and emit an "
    "image.\n\n"
    "**`dims=2d`**: consumes `fourier` (2D FT or per-frame stack), "
    "emits an image. `ny`/`nx` can force the output shape if the FT "
    "was cropped / padded (0 = auto-infer).\n\n"
    "**`dims=3d`**: consumes `fourier3d` (volume FT), emits a volume.",
    _fn_ift, manual="FT"))
