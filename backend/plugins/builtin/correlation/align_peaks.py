"""Built-in node — align.peaks (Find peaks / PK) — category: Correlation.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_find_peaks)

register(N(
    'align.peaks', 'Find peaks', 'Correlation',
    'image:image',
    ('xo:scalar,yo:scalar,pk:scalar,'
     'dx:scalar,dy:scalar,'
     'xs:scalar,ys:scalar,pks:scalar,'
     'ratio:scalar,n_found:scalar,'
     'preview:image'),
    [
        ('number_of_peaks', 'int', 1, None, 1, 100),
        ('center_origin_override', 'int', 0, None, 0, 1),
        ('new_origin_y', 'int', 0),
        ('new_origin_x', 'int', 0),
        ('peak_number_for_ratio', 'int', 1, None, 1, 100),
        ('sub_pixel', 'select', 'yes', ['yes', 'no']),
    ],
    'Locate strict local maxima in a 2D peak map.\n\n'
    '**xo, yo, pk** — top peak absolute coordinates + refined height.\n'
    '**dx, dy** — top peak offset from the image centre '
    '(nx // 2, ny // 2). Signed: negative = left/up of centre, positive '
    '= right/down. On a CC map this is exactly the (dy, dx) shift that '
    'aligns the two CC inputs; on a polar-CC map it\'s the rotation '
    'angle in pixel-column units.\n\n'
    '**xs, ys, pks** — coordinates and height of the peak selected by '
    '`peak_number_for_ratio`.\n\n'
    '**ratio** = pk / pks (a quality of the top peak relative to a '
    'chosen contender). **n_found** = how many strict local maxima '
    'were kept.\n\n'
    'When `center_origin_override = 1`, `xo`/`yo` are reported relative '
    'to `(new_origin_y, new_origin_x)` (leave both at 0 to auto-use the '
    'image centre). `dx`/`dy` are always centre-relative regardless.',
    (lambda image, number_of_peaks=1, center_origin_override=0,
            new_origin_y=0, new_origin_x=0,
            peak_number_for_ratio=1, sub_pixel='yes':
        _fn_find_peaks(image,
                       int(number_of_peaks), int(center_origin_override),
                       int(new_origin_y), int(new_origin_x),
                       int(peak_number_for_ratio), sub_pixel == 'yes')),
    manual='PK'))
