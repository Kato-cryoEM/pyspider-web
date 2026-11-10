"""Built-in node — align.cn (CN · Convolution) — category: Correlation.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_cn)

register(N('align.cn', 'Convolution', 'Correlation', 'a:image,b:image', 'image:image', [], "Real-space 2D convolution of a with b via FFT. Same output size as the inputs (mode='same'). Unlike CC, this is not sign-flipped; use CN for linear filtering with a kernel image.", _fn_cn, manual='CN'))
