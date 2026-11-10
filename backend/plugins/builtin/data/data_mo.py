"""Built-in node — data.mo (MO · Model image) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_mo)

# Which `kind` values each per-shape parameter actually applies to.
_KINDS_DENSITY   = ['blank', 'circle', 'gaussian', 'soft_circle', 'rectangle', 'ring']
_KINDS_RADIUS    = ['circle', 'gaussian', 'soft_circle', 'ring',
                    'gaussian_sum', 'gaussian_g1', 'gaussian_g2', 'gaussian_g3']
_KINDS_RADIUS_Y  = ['gaussian_sum', 'gaussian_g1', 'gaussian_g2', 'gaussian_g3']
_KINDS_WIDTH     = ['soft_circle', 'rectangle', 'ring']
_KINDS_HEIGHT    = ['rectangle']
_KINDS_CENTER    = ['circle', 'gaussian', 'soft_circle', 'rectangle', 'ring',
                    'gaussian_sum', 'gaussian_g1', 'gaussian_g2', 'gaussian_g3']
_KINDS_SEED      = ['random_uniform', 'random_gaussian']

register(N(
    'data.mo', 'Model image', 'Data', '', 'image:image',
    [
        Param('nx', 'int', 128, None, 4, 8192),
        Param('ny', 'int', 128, None, 4, 8192),
        Param('kind', 'select', 'circle', [
            'blank', 'circle', 'gaussian', 'soft_circle', 'rectangle', 'ring',
            'gaussian_sum', 'gaussian_g1', 'gaussian_g2', 'gaussian_g3',
            'random_uniform', 'random_gaussian',
            'sine_waves', 'test_sine', 'wedge',
        ]),
        Param('density', 'float', 1.0, depends_on=('kind', _KINDS_DENSITY)),
        Param('radius', 'float', 20.0, None, 0.0, 8192.0,
              depends_on=('kind', _KINDS_RADIUS)),
        Param('radius_y', 'float', 20.0, None, 0.0, 8192.0,
              depends_on=('kind', _KINDS_RADIUS_Y)),
        Param('width', 'float', 30.0, None, 0.0, 8192.0,
              depends_on=('kind', _KINDS_WIDTH)),
        Param('height', 'float', 30.0, None, 0.0, 8192.0,
              depends_on=('kind', _KINDS_HEIGHT)),
        Param('center_x', 'float', -1.0, depends_on=('kind', _KINDS_CENTER)),
        Param('center_y', 'float', -1.0, depends_on=('kind', _KINDS_CENTER)),
        Param('mean', 'float', 0.0, depends_on=('kind', 'random_gaussian')),
        Param('std', 'float', 0.35, depends_on=('kind', 'random_gaussian')),
        Param('seed', 'int', 42, depends_on=('kind', _KINDS_SEED)),
        Param('waves_csv', 'string', '1.0,0.0,1,10\n2.5,90.0,-2,5',
              depends_on=('kind', 'sine_waves'),
              placeholder='amp,phase_deg,kx,ky per line (up to 20)'),
    ],
    'Generate a synthetic 2D image. '
    'Options mirror the SPIDER manual: '
    'blank (B), circle (C), gaussian_sum (G, sum-normalized ellipse), '
    'gaussian_g1/g2/g3 (super-Gaussian masks, max=1), '
    'random_uniform / random_gaussian (R), sine_waves (S, list in waves_csv), '
    'test_sine (T, diagonal (2,2)), wedge (W, diagonal density ramp). '
    'Extras beyond SPIDER: gaussian (legacy: max=density, sigma=radius), '
    'soft_circle (flat disk of `radius` with a Gaussian-tapered edge of σ = `width`), '
    'rectangle (width × height), ring (outer radius, thickness = width). '
    'radius/radius_y = ellipse semi-axes for the G/G1/G2/G3 modes. '
    'center_x/y = -1 auto-uses the image centre. '
    'Fields below the `kind` selector show/hide in real time based on '
    'the option currently selected.',
    _fn_mo, manual='MO'))
