"""Built-in node — mask.binarize (TH B · Binarize) — category: Masking.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_binarize)

register(N('mask.binarize', 'Binarize', 'Masking', 'image:image', 'image:image', [('threshold', 'float', 0.5), ('low_value', 'float', 0.0), ('high_value', 'float', 1.0)], 'Threshold the image and replace with high_value where pixel > threshold, low_value otherwise. Useful for building masks or segmenting a foreground.', _fn_binarize, manual='TH'))
