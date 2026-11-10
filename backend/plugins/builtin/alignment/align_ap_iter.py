"""Built-in node — align.ap_iter (Iterative multi-reference alignment).

Unifies the behaviour of AP SHC (K=1 reference, iteratively updated
2D class average) with a generalisation to K>1 (K-means-style
iterative classification). Per iteration the node:

  1. Aligns every particle to the best-matching reference (shared
     polar-CC + NCC engine, identical to AP MR).
  2. Replaces each reference with the mean of the particles that
     picked it, so references sharpen as alignment improves.

K=1 is a faithful port of SPIDER AP SHC. K>1 is NOT in SPIDER's
command set — it corresponds to the AP MR + SELECT + AS R + DO
script pattern the SPIDER community typically hand-rolls — and
replaces that whole script with a single node.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _fn_ap_iter


register(N(
    "align.ap_iter", "Iterative multi-reference align", "Alignment",
    "stack:stack,references:stack",
    "aligned:stack,references:stack,assignments_csv:string,eulers_csv:string,"
    "ref_index:stack,n_matched:scalar,n_classes:scalar",
    [
        Param("n_iterations", "int", 5, min=1, max=100),
        Param("first_ring",   "int", 2, min=1, max=512),
        Param("last_ring",    "int", 15, min=1, max=512),
        Param("search_range", "int", 6, min=0, max=128),
        Param("check_mirror", "select", "yes", ["yes", "no"]),
        Param("reference_eulers_csv", "string", "", wireable=True,
              placeholder="phi,theta,psi per ref (from VO EA) — "
                          "lets eulers_csv output be BP-ready"),
    ],
    "**Iterative** multi-reference alignment with per-class reference "
    "update. Per iteration: align each particle to its best-matching "
    "reference (polar CC shortlist + NCC shift refinement), then "
    "replace each reference by the mean of the particles assigned to "
    "it. The references sharpen each iteration as alignment improves.\n\n"
    "**K = 1** (single reference, e.g. a single rough class average): "
    "the node is equivalent to SPIDER AP SHC — the one reference "
    "converges to a sharp 2D class average.\n\n"
    "**K > 1** (stack of references): K-means-style iterative "
    "classification. Each reference drifts to the mean of the "
    "particles that keep picking it. Starting from a stack of "
    "`stack.random_group_mean` (per-group means of ~N/K random "
    "particles) is a robust seed for ~4 classes.\n\n"
    "Empty classes (no particles assigned this iteration) keep their "
    "old content so they stay in the running for the next iteration.\n\n"
    "`reference_eulers_csv` (optional): one `phi,theta,psi` per "
    "reference (wire from VO EA). When set, `eulers_csv` output is "
    "populated with per-particle 3-column Euler triples suitable for "
    "`recon.bp` — the standard 3D projection-matching workflow in one "
    "node.\n\n"
    "Replaces: **AP SHC** (K=1 case), and the hand-rolled "
    "**AP MR + SELECT + AS R + DO** pattern (K>1 case).",
    _fn_ap_iter, manual="AP SHC"))
