"""Built-in node — recon.bp32f (BP 32F · Half-set reconstruction)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_bp32f)

register(N(
    'recon.bp32f', 'Half-set reconstruction', 'Reconstruction',
    'stack:stack',
    'volume:volume,half_a:volume,half_b:volume,n_a:scalar,n_b:scalar',
    [Param('eulers_csv', 'string', '', wireable=True,
           placeholder='phi,theta,psi per line (or wire from VO EA)'),
     Param('symmetry', 'select', 'c1',
           ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']),
     Param('nz', 'int', 0), Param('ny', 'int', 0), Param('nx', 'int', 0),
     Param('seed', 'int', 42)],
    'Same reconstruction algorithm as BP 3F, but also '
    'produces two independent half-set reconstructions (random 50/50 '
    'partition of the input projections). Feed `half_a` and `half_b` into '
    'an FSC computation to estimate reconstruction resolution. `seed` '
    'controls the random split so results are reproducible.',
    _fn_bp32f, manual='BP 32F'))
