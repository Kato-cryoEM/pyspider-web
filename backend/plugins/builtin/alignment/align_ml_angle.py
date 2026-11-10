"""Built-in node — align.ml_angle (ML angle estimation)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ml_angle)

register(N(
    'align.ml_angle', 'ML angle estimation', 'Alignment',
    'stack:stack,reference:volume',
    'assignments_csv:string,aligned:stack,n_frames:scalar,n_candidates:scalar,'
    'best_idx:scalar,best_phi:scalar,best_theta:scalar,best_psi:scalar,'
    'best_dy:scalar,best_dx:scalar,best_log_lik:scalar',
    [
        Param('reference_mode', 'select', 'volume', ['volume', 'stack']),
        Param('candidate_angles_csv', 'string', '', wireable=True,
              placeholder='phi,theta,psi per line (wire from VO EA)'),
        Param('sigma', 'float', 1.0, None, 0.001, 1e6),
        Param('do_shifts', 'bool', True),
        Param('search_range', 'int', 6, None, 0, 128,
              depends_on=('do_shifts', True)),
        Param('search_rotation', 'bool', True),
        Param('first_ring', 'int', 2, None, 1, 512,
              depends_on=('search_rotation', True)),
        Param('last_ring', 'int', 15, None, 1, 512,
              depends_on=('search_rotation', True)),
        Param('mask_radius', 'float', 0.0, None, 0.0, 8192.0),
    ],
    'Maximum-likelihood angle estimation. For each experimental image and '
    'each candidate Euler triple (wire from `Even Euler angles`), compute '
    'the Gaussian log-likelihood `−||y − shift(P_θ V)||² / (2σ²)`, softmax '
    'across candidates to a posterior, and report per image: MAP angle '
    'index + triple, best in-plane shift, log-likelihood at MAP, and the '
    'posterior entropy H(p) (0 = confident, log(K) = indeterminate). '
    'σ scales the noise assumption; small σ collapses the posterior to '
    'argmax (equivalent to Projection matching with SSD), large σ flattens '
    'it. `mask_radius > 0` restricts the residual to a central disk. '
    '**Comparison recipe:** wire the same volume + candidate angles through '
    '`Project volume` (batch, method=fourier) → `Multi-reference align` '
    '(AP MR) for CCC-based projection matching, then diff its `ref_index` '
    'column against `map_idx` here. Same MAP → the methods agree; differing '
    'rows are where CCC and ML disagree (typically near CTF zeros or in '
    'high-symmetry regions).',
    _fn_ml_angle))
