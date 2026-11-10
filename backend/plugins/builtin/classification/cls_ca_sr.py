"""Built-in node — cls.ca_sr (CA SR · Reconstitute frame from factors)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ca_sr)

register(N(
    'cls.ca_sr', 'Reconstitute from factors', 'Classification',
    'eigenimages:stack,mean_image:image?', 'image:image',
    [Param('scores_csv', 'string', '', wireable=True,
           placeholder='per-frame factor loadings from CA SI'),
     Param('frame_index', 'int', 0, None, 0, 100000),
     Param('factors_used', 'string', '',
           placeholder='e.g. 0,1,3   (empty = all)')],
    'Reconstitute an approximation of the original image at '
    '`frame_index` using a chosen subset of factors. Formula: '
    '`out = mean + Σ_k score[frame, k] * eigenimage_k`, summed over k in '
    '`factors_used` (or all if empty). Wire `scores_csv` from CA SI and '
    '`eigenimages` from CA SI as well.',
    _fn_ca_sr, manual='CA SR'))
