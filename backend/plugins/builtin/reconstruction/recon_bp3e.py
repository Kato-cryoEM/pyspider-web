"""Built-in node — recon.bp3e (BP 3E · Weighted BP with accumulation)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_bp3e)

register(N(
    'recon.bp3e', 'Weighted BP + accumulate', 'Reconstruction',
    'stack:stack,existing_volume:volume?', 'volume:volume',
    [Param('eulers_csv', 'string', '', wireable=True,
           placeholder='phi,theta,psi per line (or wire from VO EA)'),
     Param('symmetry', 'select', 'c1',
           ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']),
     Param('nz', 'int', 0), Param('ny', 'int', 0), Param('nx', 'int', 0)],
    'Weighted back-projection that can add its contribution '
    'to an existing volume. Wire in `existing_volume` to accumulate — useful '
    'for streaming reconstructions where projections arrive in batches. '
    'Without a prior volume it behaves the same as BP 3D.',
    _fn_bp3e, manual='BP 3E'))
