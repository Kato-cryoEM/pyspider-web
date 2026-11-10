"""Built-in node — recon.sgd_init (SGD · Ab-initio initial model) — category: Reconstruction.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_sgd_init)

register(N('recon.sgd_init', 'SGD · Ab-initio initial model', 'Reconstruction', 'stack:stack', 'volume:volume,loss_history:string,final_loss:scalar', [Param('eulers_csv', 'string', '', wireable=True), ('n_iters', 'int', 30, None, 1, 5000), ('batch_size', 'int', 10, None, 1, 4096), ('lr', 'float', 0.05, None, 1e-06, 10.0), ('seed', 'int', 0, None, 0, 100000)], '**Not a SPIDER command — modern cryo-SPARC/RELION-style ab-initio 3D reconstruction via stochastic gradient descent.** Initialises the volume with Gaussian noise (σ=0.01), then per iteration picks a random mini-batch of `batch_size` particles, forward-projects the current V at each of their angles, computes the L2 residual against the experimental image, and subtracts the back-projected residual (the gradient) from V. If `eulers_csv` is empty, uniform-random angles are used — that only makes sense as a stress test; for a meaningful reconstruction wire in known/estimated poses (from VO EA for a synthetic demo, or from projection matching). Watch `final_loss` decrease over iterations; the returned `loss_history` is a comma-separated list you can plot.', _fn_sgd_init))
