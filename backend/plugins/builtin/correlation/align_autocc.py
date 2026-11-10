"""Built-in node — align.autocc (AC · Auto-correlation) — category: Correlation.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_autocorrelate)

register(N('align.autocc', 'Auto-correlation', 'Correlation', 'image:image', 'image:image', [], 'Self-correlation of an image (CC with itself). Peaks reveal periodic structure; the central peak measures total signal power.', _fn_autocorrelate, manual='AC'))
