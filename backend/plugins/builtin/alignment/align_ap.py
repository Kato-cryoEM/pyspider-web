"""Built-in node — align.ap (Unified single/multi-reference alignment).

Consolidates the following legacy nodes into one general-purpose
alignment node:

  AP MR / AP MQ (rot + shift, multi-reference)
  AP MH / AP MD (rotation only, multi-reference)
  AP SH          (single reference, parameters only — K=1 of AP MR)
  AP C / AP CA   (rotational align + classify — ref_index IS the class)
  AP CM          (full align + classify — same as AP MR with ref_index)
  OR SH          (single pair — K=1 stack of length 1)
  align.rotation (rotation-only pair — search_range=0 + length-1 stack)

Behaviour is controlled by three plain parameters instead of a
``method`` dropdown or a node-per-combination:

  search_range = 0  →  rotation only (= AP MH / MD / rotation)
  search_range > 0  →  rotation + shift (= AP MR / MQ / OR SH)
  check_mirror=yes  →  also test flip_h(particle)
  references of length 1  →  single-reference alignment (= AP SH / OR SH)
  references of length K  →  multi-reference, ref_index gives class

Outputs the same scalar pack as AP MR + a BP-ready 3-column
``eulers_csv`` when ``reference_eulers_csv`` is wired.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _multi_ref_align


def _fn_ap(stack, references,
           first_ring: int = 2, last_ring: int = 15,
           search_range: int = 6, check_mirror: str = "yes",
           reference_eulers_csv: str = ""):
    return _multi_ref_align(
        stack, references,
        int(first_ring), int(last_ring), int(search_range),
        do_shift=(int(search_range) > 0),
        check_mirror=(check_mirror == "yes"),
        reference_eulers_csv=reference_eulers_csv,
    )


_OUT = (
    "aligned:stack,assignments_csv:string,eulers_csv:string,ref_index:stack,"
    "n_matched:scalar,best_ref_idx:scalar,best_peak:scalar,"
    "best_angle_deg:scalar,best_dy:scalar,best_dx:scalar"
)


register(N(
    "align.ap", "Align to reference(s)", "Alignment",
    "stack:stack,references:stack", _OUT,
    [
        Param("first_ring",   "int", 2, min=1, max=512),
        Param("last_ring",    "int", 15, min=1, max=512),
        Param("search_range", "int", 6, min=0, max=128),
        Param("check_mirror", "select", "yes", ["yes", "no"]),
        Param("reference_eulers_csv", "string", "", wireable=True,
              placeholder="phi,theta,psi per ref (from VO EA) — "
                          "enables 3-column eulers_csv output for BP"),
    ],
    "Unified alignment of a particle stack to one or more references.\n\n"
    "**Behaviour is controlled by parameters, not node choice:**\n"
    "  • `references` length 1 → single-reference alignment (classic class-"
    "averaging template fit, same as SPIDER AP SH / OR SH).\n"
    "  • `references` length K>1 → multi-reference. Each particle picks "
    "its best-matching ref; `ref_index` output records which one (= "
    "class label, no separate classify node needed).\n"
    "  • `search_range = 0` → rotation only (= SPIDER AP MH / AP MD / "
    "align.rotation).\n"
    "  • `search_range > 0` → rotation + XY shift (= AP MR / MQ / OR SH).\n"
    "  • `check_mirror = yes` → also test the horizontally-flipped "
    "particle; AP MD-style workflows use `no` once handedness is fixed.\n\n"
    "A single particle (2D image) is auto-wrapped to a length-1 stack, "
    "so the same node works for the SPIDER OR SH use case.\n\n"
    "**Reconstruction bridge:** wire VO EA's `eulers_csv` (one "
    "`phi,theta,psi` per reference) into the `reference_eulers_csv` "
    "param. Each matched particle then gets a composed `phi,theta,psi` "
    "on the `eulers_csv` output ready for `recon.bp` or `recon.pj3q`.\n\n"
    "Replaces: AP MR, AP MQ, AP MH, AP MD, AP SH, AP C, AP CA, AP CM, "
    "OR SH, align.rotation.",
    _fn_ap, manual="AP MR"))
