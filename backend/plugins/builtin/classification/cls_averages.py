"""Built-in node — cls.averages (AS R · Class averages) — category: Classification.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_class_avg)

register(N('cls.averages', 'Class averages', 'Classification', 'stack:stack,labels:image', 'stack:stack', [('n_classes', 'int', 4, None, 2, 64)], 'Compute the per-class mean image. Feed the raw image stack and the labels output from CL KM; the result is a stack of n_classes averages (one per cluster). Standard next-step after 2D classification.', _fn_class_avg, manual='AS R'))
