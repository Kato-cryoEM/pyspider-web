"""Built-in nodes — Filter LP / HP, SPIDER FQ family consolidated.

Previously this file registered ten nodes (tophat / gauss / fermi / butter /
raised-cos × LP / HP). The ten shared the exact same I/O signature
(image → image); only their frequency response differed. They are now
consolidated into two nodes — `filter.lp` and `filter.hp` — each with a
`kind` dropdown that picks the response shape and `depends_on`-gated
parameters so only the fields relevant to the chosen kind are shown.

Rationale: ten palette entries for one algorithm family (apply a radial
window in Fourier space) made side-by-side comparison awkward (replace
node + rewire) and bloated the palette. One node with a dropdown lets a
tutorial compare TopHat vs Gaussian by changing one parameter.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param, d
from backend.nodes import h, filter_image


# Dropdown values map 1:1 onto the SPIDER FQ kinds that `filter_image`
# understands; the direction suffix (-lp / -hp) is appended at call time.
_KINDS = ["tophat", "gauss", "fermi", "butter", "raised_cos"]

_FC_KINDS   = ["tophat", "gauss", "fermi"]       # use `fc`
_BAND_KINDS = ["butter", "raised_cos"]           # use `fp` + `fs`


def _call_filter(image, direction: str, kind: str, fc: float, temp: float,
                 fp: float, fs: float, padding: str):
    full_kind = f"{kind.replace('_', '-')}-{direction}"
    a = h(image)
    if kind in ("tophat", "gauss"):
        out = filter_image(a, full_kind, fc=float(fc), padding=padding)
    elif kind == "fermi":
        out = filter_image(a, full_kind, fc=float(fc), temp=float(temp),
                           padding=padding)
    else:  # butter / raised_cos
        out = filter_image(a, full_kind, fp=float(fp), fs=float(fs),
                           padding=padding)
    return {"image": d(np.asarray(out, dtype=np.float32))}


def _make_filter(direction: str) -> None:
    """direction = 'lp' (low-pass) or 'hp' (high-pass)."""
    assert direction in ("lp", "hp")
    is_lp = direction == "lp"
    nid = f"filter.{direction}"
    label = "Filter · Low-pass" if is_lp else "Filter · High-pass"

    pass_desc = "passes" if is_lp else "removes"
    stop_desc = "cuts above" if is_lp else "below"

    doc = (
        f"Apply a radial frequency filter in Fourier space. The `kind` "
        f"dropdown picks the response shape:\n\n"
        f"  • `tophat`      — sharp rectangular cutoff at `fc`. "
        f"Strongest frequency selectivity, but introduces ringing in real "
        f"space (Gibbs phenomenon at sharp edges).\n"
        f"  • `gauss`       — Gaussian attenuation `exp(-(f/fc)²/2)`. "
        f"Smooth, no ringing. The default go-to for cryo-EM pre-filters.\n"
        f"  • `fermi`       — sigmoid centred at `fc`; edge sharpness "
        f"controlled by `temp` (small = near-tophat, large = near-Gaussian).\n"
        f"  • `butter`      — Butterworth response; order is auto-fitted "
        f"from the (`fp`, `fs`) passband / stopband pair.\n"
        f"  • `raised_cos`  — cosine roll-off across the [`fp`, `fs`] "
        f"transition band.\n\n"
        f"Frequencies are in cycles per pixel (0.5 = Nyquist). "
        f"`padding='border'` extends the image with the perimeter mean "
        f"before FFT (SPIDER's default); `'none'` reproduces FQ NP. "
        f"The direction of this node is **{direction.upper()}**: it "
        f"{pass_desc} frequencies, {stop_desc} fc / fs."
    )

    def fn(image, kind: str = "gauss", fc: float = 0.1, temp: float = 0.02,
           fp: float = 0.10, fs: float = 0.15, padding: str = "border"):
        return _call_filter(image, direction, kind, fc, temp, fp, fs, padding)

    params = [
        Param("kind", "select", "gauss", _KINDS),
        # fc is visible for tophat / gauss / fermi
        Param("fc", "float", 0.1, min=0.0, max=0.5,
              depends_on=("kind", _FC_KINDS)),
        # temp is only meaningful for fermi
        Param("temp", "float", 0.02, min=0.001, max=0.5,
              depends_on=("kind", "fermi")),
        # fp / fs are visible for butter / raised_cos
        Param("fp", "float", 0.10, min=0.0, max=0.5,
              depends_on=("kind", _BAND_KINDS)),
        Param("fs", "float", 0.15, min=0.0, max=0.5,
              depends_on=("kind", _BAND_KINDS)),
        Param("padding", "select", "border", ["border", "none"]),
    ]
    register(N(nid, label, "Filter",
               "image:image", "image:image",
               params, doc, fn, manual="FQ"))


_make_filter("lp")
_make_filter("hp")
