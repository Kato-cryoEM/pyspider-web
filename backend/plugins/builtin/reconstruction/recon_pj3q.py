"""Built-in node — recon.pj3q (Project volume) — category: Reconstruction.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_pj3q)

register(N(
    'recon.pj3q', 'Project volume', 'Reconstruction',
    'volume:volume', 'stack:stack',
    [
        Param('batch', 'bool', True),
        Param('eulers_csv', 'string', '', wireable=True,
              depends_on=('batch', True),
              placeholder='phi,theta,psi per line (or wire from VO EA)'),
        Param('phi', 'float', 0.0, min=-360.0, max=360.0, step=1.0,
              depends_on=('batch', False)),
        Param('theta', 'float', 0.0, min=-360.0, max=360.0, step=1.0,
              depends_on=('batch', False)),
        Param('psi', 'float', 0.0, min=-360.0, max=360.0, step=1.0,
              depends_on=('batch', False)),
        Param('method', 'select', 'fourier', ['fourier', 'real'],
              depends_on=('batch', True)),
    ],
    "Project a 3D volume along a viewing direction defined by "
    "ZYZ Euler angles (phi around Z, theta around Y, psi around Z; degrees). "
    "**batch = ON (default)**: `eulers_csv` supplies one 'phi,theta,psi' "
    "triple per line (wire from VO EA), outputting a stack of that many "
    "projections. **batch = OFF**: manual `phi/theta/psi` inputs; output is "
    "a length-1 stack. Use `stack.index` downstream if you need the single "
    "image. **method** (batch only): `fourier` (default) computes one 3D FFT "
    "of the volume and extracts oriented central slices per angle — "
    "~20-50× faster than `real` for dense batches at box ≥ 128.",
    _fn_pj3q, manual='PJ 3Q'))
