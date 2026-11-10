"""Built-in node — stack.filter (Stack · FQ filter each frame) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_FQ, _fn_stack_filter)

register(N('stack.filter', 'Filter each frame', 'Stack', 'stack:stack', 'stack:stack', [('kind', 'select', 'gauss-lp', [k for k, _, _, _ in _FQ]), ('fc', 'float', 0.1, None, 0.0, 0.5), ('temp', 'float', 0.02, None, 0.001, 0.5), ('fp', 'float', 0.1, None, 0.0, 0.5), ('fs', 'float', 0.15, None, 0.0, 0.5), ('padding', 'select', 'border', ['border', 'none'])], 'DO-loop analog — apply an FQ filter to every frame of a stack. fc is used by sharp / Gaussian kinds, temp by Fermi, and (fp, fs) by Butterworth / raised-cosine (only the relevant knobs matter per kind).', _fn_stack_filter))
