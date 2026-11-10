"""Built-in node — recon.bpmem2 (BP MEM 2 · Maximum entropy 2D BP)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_bpmem2)

register(N(
    'recon.bpmem2', 'MaxEnt 2D reconstruction', 'Reconstruction',
    'stack:stack', 'image:image',
    [Param('eulers_csv', 'string', '', wireable=True,
           placeholder='phi,theta,psi per line (psi = in-plane angle)'),
     Param('iterations', 'int', 30, None, 1, 1000),
     Param('prior_strength', 'float', 1.0, None, 0.01, 100.0),
     Param('ny', 'int', 0), Param('nx', 'int', 0)],
    'SPIDER BP MEM 2 — 2D reconstruction from projections via a simple '
    'multiplicative Maximum Entropy scheme (Cornwell–Evans style). Uses the '
    'central row of each input frame as a 1D shadow at angle `psi`, then '
    'iteratively adjusts a positive current estimate to match all shadows. '
    '`iterations` = number of update steps; `prior_strength` biases toward a '
    'flat solution (larger = smoother, slower to converge). Educational / '
    'exploratory implementation; use BP 3F / BP CG for production 3D work.',
    _fn_bpmem2, manual='BP MEM 2'))
