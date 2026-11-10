"""Built-in node — data.betagal (Test · β-gal) — category: Data.

Loads a bundled β-galactosidase image or stack from `backend/bundled_data/
betagal/`. Data is derived from the publicly-distributed RELION 4.0
tutorial dataset (courtesy of the RELION consortium and the Namba lab).
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_betagal)

# Two output ports whose visibility depends on `variant` — average /
# raw_particle expose ▸ image (2D); stack_100 exposes ▸ stack (3D).
# The runtime always populates exactly one of the two so wiring the wrong
# port isn't possible when only the matching one is shown.
_OUTPUTS = [
    Port(name='image', kind='image',
         depends_on=('variant', ['average', 'raw_particle'])),
    Port(name='stack', kind='stack',
         depends_on=('variant', 'stack_100')),
    Port(name='volume', kind='volume',
         depends_on=('variant', 'map3d')),
]

register(N(
    'data.betagal', 'Test · β-gal', 'Data',
    '', _OUTPUTS,
    [
        Param('variant', 'select', 'average',
              ['average', 'raw_particle', 'stack_100', 'map3d']),
        Param('class_index', 'int', 38, None, -1, 49,
              depends_on=('variant', 'average')),
        Param('size', 'int', 128, None, 16, 1024),
    ],
    'Load a bundled β-galactosidase image / stack / volume. **variant**: '
    '`average` (one 2D class average, pick which via `class_index`); '
    '`raw_particle` (one un-averaged extracted particle — realistic '
    'cryo-EM SNR); `stack_100` (100 raw particles as a 3D stack for '
    'AS S / AP MR / cls.pca demos); **`map3d`** (the REAL β-galactosidase '
    '3D reconstruction from the RELION 3.0 tutorial reference, native '
    '128³ @ 1 Å/px, soft-cosine low-passed at 5 Å then cubic-spline '
    'resampled to `size × size × size` — feed into the 3D viewer, '
    'BP · Back-project, PJ 3Q, etc.).\n\n'
    'The right-hand output port switches automatically with the variant: '
    '▸ image (2D) for `average`/`raw_particle`, ▸ stack (3D) for '
    '`stack_100`, ▸ volume (3D) for `map3d`. Only the active port is '
    'populated at run time — the others are hidden so you cannot wire '
    'the wrong one by accident.\n\n'
    '**class_index** (average only, default **38**): which of the 50 '
    'RELION Class2D averages to load. Set to `-1` to auto-cycle — each '
    'Run then advances through the classes (0 → … → 49 → 0). Any value '
    '0–49 pins that specific class.\n\n'
    '**size** resizes each frame / volume: 2D and stack use Fourier crop '
    '(smaller than native) or cubic B-spline zoom (larger); `map3d` '
    'generates the density directly at `size × size × size` — set to '
    '**256** for the tutorial default. Native 2D: 64 (average), 256 '
    '(raw_particle, stack_100).',
    _fn_betagal))
