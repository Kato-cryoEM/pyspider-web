"""Built-in node — ft.forward (Forward FFT, 2D/3D) — Fourier.

Unified 2D / 3D forward FFT. The ``dims`` dropdown defaults to ``auto``,
which chooses based on the input's dimensionality: 2D → 2D FFT, 3D → 3D
FFT. Override to ``2d`` when a 3D array is really a stack of 2D frames
and you want per-frame FFT.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, Port
from backend.nodes import (_fn_ft)


register(N(
    "ft.forward", "Forward FFT", "Fourier",
    "image:image",
    [
        Port("fourier",   "fourier",   depends_on=("dims", ["auto", "2d"])),
        Port("fourier3d", "fourier3d", depends_on=("dims", ["auto", "3d"])),
    ],
    [Param("dims", "select", "auto", ["auto", "2d", "3d"])],
    "Forward FFT with DC pre-shifted to the geometric centre.\n\n"
    "**`dims=auto`** (default): 2D input → 2D FFT; 3D input → full 3D FFT. "
    "Only override when a 3D array is actually a stack of 2D frames.\n\n"
    "**`dims=2d`**: numpy fft2 on the last two axes. A 2D image gives a "
    "single 2D FT; a 3D stack `(N, ny, nx)` gives per-frame 2D FTs "
    "(PJ 3Q → FT → MU → IFT flows unchanged). Output appears on the "
    "`fourier` port (kind=fourier).\n\n"
    "**`dims=3d`**: numpy fftn over all three axes — a proper volumetric "
    "FT. Requires a 3D volume input. Output appears on `fourier3d` "
    "(kind=fourier3d) so 2D-only downstream nodes don't accidentally "
    "consume a volume FT.\n\n"
    "`ft.inverse` matches so FT → IFT is identity.",
    _fn_ft, manual="FT"))
