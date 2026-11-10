"""Built-in node — mask.threshold (TH · Clip range) — category: Masking.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_threshold)

register(N('mask.threshold', 'Clip range', 'Masking', 'image:image', 'image:image', [('direction', 'select', 'both', ['above', 'below', 'both']), ('upper', 'float', 1.0), ('lower', 'float', 0.0)], 'SPIDER TH direction: above (A) clips values above `upper` down to `upper`; below (B) clips values below `lower` up to `lower`; both (C) clips both sides into [lower, upper].', _fn_threshold, manual='TH'))
