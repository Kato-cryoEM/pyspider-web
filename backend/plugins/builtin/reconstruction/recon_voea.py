"""Built-in node — recon.voea (VO EA · Even Euler angles) — category: Reconstruction.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_voea)

register(N(
    'recon.voea', 'Even Euler angles', 'Reconstruction',
    '', 'eulers_csv:string,num_angles:scalar',
    [
        Param('delta_theta', 'float', 15.0, None, 0.0, 90.0, step=1.0),
        Param('theta_min', 'float', 0.0, None, 0.0, 180.0, step=1.0),
        Param('theta_max', 'float', 90.0, None, 0.0, 180.0, step=1.0),
        Param('phi_min', 'float', 0.0, None, 0.0, 360.0, step=1.0),
        Param('phi_max', 'float', 360.0, None, 0.0, 360.0, step=1.0),
        Param('symmetry', 'select', 'c1',
              ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
               'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']),
    ],
    "Quasi-even Eulerian angles on a spherical patch. "
    "Standard input for BP 3F: wire eulers_csv into BP 3F's eulers_csv param. "
    "`delta_theta` sets the angular density; phi step per theta band is "
    "adjusted by sin(theta) to keep uniform density on the sphere. "
    "`symmetry` restricts the output to the asymmetric unit of the chosen "
    "point group — use the **same value** on the downstream BP 3F / BP CG / "
    "BP 32F, which then replicates the projections around the symmetry axes "
    "during reconstruction. c1 = full sphere, cN = phi ∈ [0, 360/N), dN = "
    "cN + upper hemisphere (theta ∈ [0, 90]).",
    _fn_voea, manual='VO EA'))
