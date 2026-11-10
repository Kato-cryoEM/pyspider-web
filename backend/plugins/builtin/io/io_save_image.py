"""Built-in node — io.save_image (Save Image · PNG / JPEG / TIFF / BMP).

Distinct from io.save (MRC, lossless float32): this node produces a
display-ready 8-bit or 16-bit image via Pillow, so any viewer can open it.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_save_image)

register(N(
    'io.save_image', 'Save Image', 'I/O', 'image:image', '',
    [
        Param('path', 'string', '',
              placeholder='out.png  (use {i} for DO iterations)'),
        Param('format', 'select', 'auto',
              ['auto', 'png', 'jpeg', 'tiff', 'bmp']),
        Param('bit_depth', 'select', '8', ['8', '16']),
        Param('normalize', 'select', 'auto', ['auto', 'fixed', 'none']),
        Param('min_value', 'float', 0.0,
              depends_on=('normalize', 'fixed')),
        Param('max_value', 'float', 1.0,
              depends_on=('normalize', 'fixed')),
        Param('invert', 'bool', False),
        Param('index', 'int', -1),
    ],
    'Save the incoming image as a viewer-friendly file (PNG / JPEG / TIFF / '
    'BMP) via Pillow. The float array is normalised and quantised to 8 or '
    '16 bits — use `io.save` (Save MRC) when you need lossless float32. '
    'normalize=auto stretches [min, max] of the data to full range; '
    'normalize=fixed maps [min_value, max_value] instead; normalize=none '
    'assumes data is already in [0, 1] and just clips. '
    'JPEG / BMP are 8-bit only, so a 16-bit request is downgraded silently. '
    '{i} in the path is replaced by the wired DO-loop iteration index.',
    _fn_save_image))
