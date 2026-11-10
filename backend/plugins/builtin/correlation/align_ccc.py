"""Built-in node — align.ccc (CC C · Correlation coeff) — category: Correlation.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ccc)

register(N('align.ccc', 'Correlation coeff', 'Correlation', 'a:image,b:image,mask:image?', 'ccc:scalar,av1:scalar,sd1:scalar,av2:scalar,sd2:scalar,eud:scalar', [], 'Pearson correlation coefficient plus per-image mean/std and the Euclidean distance between the images. Optional mask input restricts the calculation to pixels where mask > 0.', _fn_ccc, manual='CCC'))
