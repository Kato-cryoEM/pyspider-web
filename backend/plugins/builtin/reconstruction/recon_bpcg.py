"""Built-in node — recon.bpcg (BP CG · CG reconstruction) — category: Reconstruction.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_bp_cg)

register(N('recon.bpcg', 'CG reconstruction', 'Reconstruction', 'stack:stack', 'volume:volume,info:scalar,iters:scalar', [Param('eulers_csv', 'string', '', wireable=True, placeholder='phi,theta,psi per line (or wire from VO EA)'), ('iterations', 'int', 20, None, 1, 500), ('regularization', 'float', 0.01, None, 0.0, 100.0), ('radius', 'int', 0, None, 0, 8192)], "Solves (P^T P + λ I) V = P^T Y using scipy's conjugate gradient on the normal equations. iterations bounds the CG steps; regularization is Tikhonov λ (larger = smoother, more damped); radius > 0 restricts the output to a sphere of that radius. Only supports square projections (ny = nx). info is the scipy CG return code (0 = converged).", _fn_bp_cg, manual='BP CG'))
