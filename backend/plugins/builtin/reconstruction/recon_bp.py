"""Built-in node — recon.bp (back-project / Fourier insert).

Two-dimensional AND three-dimensional in a single node — the output
dimensionality is **auto-detected from the stack shape**:

  stack shape (N, 1, W)   → inverse 2D Radon transform  → `image` output
  stack shape (N, H, W)   → 3D reconstruction           → `volume` output
  with H > 1

Four reconstruction `method` branches, same semantics as the SPIDER
BP 3F / BP 3D / BP 3G / BP 3N commands, adapted to the current
dimension. `recon.bp32f` is kept as a separate node because it emits
half-set volumes (`half_a`, `half_b`, `n_a`, `n_b`) in addition to the
main volume.
"""

from __future__ import annotations

import numpy as np

from backend.plugin_api import register, N, Param, Port, d
from backend.nodes import (
    _fn_bp3d, _fn_bp3f, _fn_bp3g, _fn_bp3n,
    _bp_2d_real, _bp_2d_fourier, h,
)


_METHODS = ["bp3f", "bp3d", "bp3g", "bp3n"]

_FN_3D = {
    "bp3d": _fn_bp3d,
    "bp3f": _fn_bp3f,
    "bp3g": _fn_bp3g,
    "bp3n": _fn_bp3n,
}

_METHOD_DESC = {
    "bp3f": "Weighted back-projection (SPIDER BP 3F). Standard "
            "real-space back-projection with a ramp filter. The usual "
            "default. In 2D: filtered back-projection (classical FBP).",
    "bp3d": "Direct (unfiltered) back-projection (SPIDER BP 3D). Same "
            "voxel-smearing kernel as BP 3F but **without** the ramp "
            "filter — reconstructions have the classical low-pass bias. "
            "In 2D: raw unfiltered 2D back-projection.",
    "bp3g": "Fourier gridding (SPIDER BP 3G). Each central-slice sample "
            "is splatted onto a small tri-linear neighbourhood of the "
            "3D (or 2D) FT; weights are divided out, inverse FFT "
            "recovers the volume (or image). Smoother than NN "
            "insertion at sparse angular sampling.",
    "bp3n": "Fourier nearest-neighbour insertion (SPIDER BP 3N). "
            "Fastest Fourier variant; samples are inserted onto the "
            "nearest FT voxel. Sparse angular sampling causes "
            "striations.",
}


def _parse_phi(eulers_csv: str, n: int):
    """Return (n,) array of phi values from a `phi,theta,psi` CSV. The
    2D case only uses phi; missing / short rows default to 0.0."""
    rows = [r for r in eulers_csv.strip().splitlines() if r.strip()]
    out = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rows[:n]):
        parts = r.split(",")
        if parts and parts[0].strip():
            try:
                out[i] = float(parts[0])
            except ValueError:
                out[i] = 0.0
    return out


def _fn_bp(stack, method: str = "bp3f",
           eulers_csv: str = "", symmetry: str = "c1",
           nz: int = 0, ny: int = 0, nx: int = 0):
    s = h(stack)
    # ------------------------------------------------------------------
    # Auto-detect dimensionality from the stack shape.
    #   (N, 1, W)       → 1D projections → 2D reconstruction (image)
    #   (N, H, W), H>1  → 2D projections → 3D reconstruction (volume)
    # ------------------------------------------------------------------
    if s.ndim == 3 and s.shape[1] == 1:
        # 2D inverse Radon case.
        s_1d = s[:, 0, :]
        n, W = s_1d.shape
        angles = _parse_phi(eulers_csv, n)
        N = int(nx) if nx else W
        if method in ("bp3f", "bp3d"):
            img = _bp_2d_real(s_1d, angles, N, ramp_filter=(method == "bp3f"))
        elif method == "bp3g":
            img = _bp_2d_fourier(s_1d, angles, N, gridding=True)
        elif method == "bp3n":
            img = _bp_2d_fourier(s_1d, angles, N, gridding=False)
        else:
            raise ValueError(f"unknown method: {method!r}. Known: {_METHODS}")
        return {"image": d(img)}

    # 3D case.
    fn = _FN_3D.get(method)
    if fn is None:
        raise ValueError(f"unknown method: {method!r}. Known: {_METHODS}")
    return fn(stack, eulers_csv=eulers_csv, symmetry=symmetry,
              nz=nz, ny=ny, nx=nx)


_DOC = (
    "Reconstruction from a projection stack and per-frame angles. "
    "The dimensionality is **auto-detected from the stack shape** — a "
    "`(N, 1, W)` stack flows through the 2D path (inverse Radon → "
    "`image` output); anything else flows through the 3D path "
    "(→ `volume` output). The `method` dropdown picks the algorithm "
    "within either dimension:\n\n"
    + "\n".join(f"  • `{m}` — {_METHOD_DESC[m]}" for m in _METHODS)
    + "\n\n"
    "`eulers_csv` has one `phi,theta,psi` triple per line (2D case uses "
    "only the first column — the in-plane projection angle). Accepts a "
    "wire from `data.sim.projections` directly. `symmetry` applies Cn "
    "cyclic symmetry around the Z axis by adding rotated copies of each "
    "projection (3D only). `nz`/`ny`/`nx` set the output size — use 0 "
    "to default to a cube (3D) or `W × W` (2D using `nx`).\n\n"
    "By the central-slice theorem, real-space back-projection and "
    "Fourier insertion are mathematically equivalent — the methods "
    "here differ in implementation (and numerical artefact pattern), "
    "not in the underlying reconstruction mapping."
)


register(N(
    "recon.bp", "Back-project / Fourier insert", "Reconstruction",
    "stack:stack",
    [
        Port("volume", "volume"),
        Port("image",  "image"),
    ],
    [Param("method", "select", "bp3f", _METHODS),
     Param("eulers_csv", "string", "", wireable=True,
           placeholder="phi,theta,psi per line (or wire from VO EA)"),
     Param("symmetry", "select", "c1",
           ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]),
     Param("nz", "int", 0),
     Param("ny", "int", 0),
     Param("nx", "int", 0)],
    _DOC, _fn_bp, manual="BP 3F"))
