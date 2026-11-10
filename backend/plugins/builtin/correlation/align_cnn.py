"""Built-in node — align.cnn (CN N · Normalized convolution) — category: Correlation.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_cnn)

register(N('align.cnn', 'Normalized convolution', 'Correlation', 'a:image,b:image', 'image:image', [], 'Normalized convolution: numerator is the FFT convolution of a and b, denominator is sqrt(local energy of a in the support of b) times ||b||₂. Analogous to normalized cross-correlation and used in template matching.', _fn_cnn, manual='CN N'))
