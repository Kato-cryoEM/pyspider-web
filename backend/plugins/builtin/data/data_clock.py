"""Built-in node — data.clock (Test · Clock face) — category: Data.

Clock face with hour ticks, radially-oriented digits at 12/3/6/9, and
hour + minute hands at the (hour, minute) time. Great for demoing the
polar transform — rotating the image shifts every clock feature
horizontally by the same number of columns in polar space; the
asymmetric hand layout pins down the absolute rotation.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param
from backend.nodes import _clock_image, d


def _fn(size: int = 256, hour: int = 10, minute: int = 10):
    return {"image": d(_clock_image(int(size), int(hour), int(minute)))}


register(N(
    'data.clock', 'Test · Clock face', 'Data',
    '', 'image:image',
    [Param('size',   'int', 128, min=64, max=1024),
     Param('hour',   'int',  10, min=0,  max=23),
     Param('minute', 'int',  10, min=0,  max=59)],
    'Clock-face test image: bezel + 12 hour ticks (long at 12/3/6/9) + '
    'radially-oriented digits **12, 3, 6, 9** (top of each glyph points '
    'AWAY from centre) + hour and minute hands. Default time 10:10 is '
    'the classic clock-ad pose — the two hands are widely separated. '
    'Rotationally-structured features (ticks, digits) make polar '
    'transform + shift alignment demos crystal-clear.',
    _fn))
