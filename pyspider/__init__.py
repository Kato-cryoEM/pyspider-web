"""pyspider — modern Python access to SPIDER single-particle analysis data.

The original SPIDER package (Frank et al., 1996, J. Struct. Biol. 116:190)
is preserved in /Applications/spider as the canonical Fortran reference.
This package provides:

  - SPIDER image / volume / stack file I/O                  (pyspider.format)
  - SPIDER document file I/O                                (TODO)
  - thin wrappers around the modernized arm64 binary        (TODO)
  - educational notebooks pairing SPIDER algorithms with    (../notebooks)
    NumPy reimplementations.

Status: I/O for simple 2D images / 3D volumes / image stacks.
"""

from .format import (
    SpiderHeader,
    read_header,
    read_image,
    read_stack,
    read_fourier,
    write_image,
    write_fourier,
    inspect,
)
from . import align, classify, ctf, filters, reconstruct, simulate, testdata


def use_japanese_font(font_name: str | None = None) -> str:
    """Configure matplotlib to render Japanese text without warnings.

    Tries the given ``font_name`` first, then falls back through a list
    of CJK-capable fonts that ship with macOS / common Linux setups.

    Call this once per notebook (after ``import matplotlib.pyplot``) to
    silence the "Glyph XXXXX missing from font" warnings that DejaVu Sans
    spews when titles contain Japanese.

    Returns the font name that was actually selected.
    """
    import matplotlib
    import matplotlib.font_manager as _fm

    candidates = [font_name] if font_name else []
    candidates += [
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "Hiragino Mincho ProN",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "Noto Sans CJK",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    picked = next((c for c in candidates if c and c in available), None)
    if picked is None:
        return matplotlib.rcParams["font.family"][0]
    matplotlib.rcParams["font.family"] = picked
    matplotlib.rcParams["axes.unicode_minus"] = False
    return picked
from .filters import filter_image, filter_response
from .align import (
    cross_correlate,
    find_peaks,
    parabolic_refine,
    polar_sample,
    rotational_angle,
    rotational_correlate,
    translational_shift,
)

__all__ = [
    "SpiderHeader",
    "read_header",
    "read_image",
    "read_stack",
    "read_fourier",
    "write_image",
    "write_fourier",
    "inspect",
    "filters",
    "filter_image",
    "filter_response",
    "align",
    "classify",
    "ctf",
    "reconstruct",
    "simulate",
    "testdata",
    "use_japanese_font",
    "cross_correlate",
    "find_peaks",
    "parabolic_refine",
    "polar_sample",
    "rotational_angle",
    "rotational_correlate",
    "translational_shift",
]

__version__ = "0.1.0"
