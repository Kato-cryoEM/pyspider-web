"""Built-in node — align.ap_free (Reference-free iterative alignment).

Consolidates SPIDER AP RA / AP SA / AP SR into one node — the
underlying ``_ref_free_align`` engine already had ``do_rotation``
and ``do_shift`` flags; the new node simply exposes them as
user-visible parameters so a dropdown is no longer required.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _ref_free_align


def _fn_ap_free(stack,
                n_iterations: int = 5,
                do_rotation: bool = True,
                do_shift: bool = True,
                first_ring: int = 2,
                last_ring: int = 15,
                search_range: int = 6,
                check_mirror: str = "yes"):
    return _ref_free_align(
        stack, int(n_iterations),
        bool(do_rotation), bool(do_shift),
        int(first_ring), int(last_ring),
        int(search_range),
        check_mirror == "yes",
    )


register(N(
    "align.ap_free", "Reference-free iterative align", "Alignment",
    "stack:stack",
    "aligned:stack,reference:image,n_iterations:scalar",
    [
        Param("n_iterations", "int", 5, min=1, max=100),
        Param("do_rotation", "bool", True),
        Param("do_shift",    "bool", True),
        Param("first_ring",  "int", 2, min=1, max=512,
              depends_on=("do_rotation", True)),
        Param("last_ring",   "int", 15, min=1, max=512,
              depends_on=("do_rotation", True)),
        Param("search_range", "int", 6, min=1, max=128,
              depends_on=("do_shift", True)),
        Param("check_mirror", "select", "yes", ["yes", "no"],
              depends_on=("do_rotation", True)),
    ],
    "Reference-free iterative alignment. No external template is "
    "needed — each iteration uses the current stack mean as the "
    "reference, aligns every frame to it, re-averages, and repeats. "
    "The converged reference is the result of the ensemble averaging "
    "out whatever structure the particles agree on.\n\n"
    "**Mode is controlled by two bool flags:**\n"
    "  • `do_rotation`, `do_shift` both ON → full alignment (= SPIDER "
    "AP SR).\n"
    "  • `do_rotation=yes`, `do_shift=no` → rotation only (= AP RA).\n"
    "  • `do_rotation=no`, `do_shift=yes` → shift only (= AP SA).\n\n"
    "Replaces: AP RA, AP SA, AP SR.",
    _fn_ap_free, manual="AP SR"))
