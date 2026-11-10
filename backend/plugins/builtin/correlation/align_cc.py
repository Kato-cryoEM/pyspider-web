"""Built-in node — align.cc (CC · Cross-correlate) — category: Correlation.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_cc)

register(N('align.cc', 'Cross-correlate', 'Correlation',
           'a:image,b:image', 'cc:image',
           [('log_display', 'bool', True)],
           'Zero shift is placed at the centre of the output — a peak at '
           '(ny//2 + dy, nx//2 + dx) means shifting b by (dy, dx) aligns '
           'it with a. Feed the output to PK · Find peaks to read the '
           'shift. `log_display` (default on) applies `sign(cc)·log1p(|cc|)` '
           'to compress the map for display so side peaks are visible; '
           'PEAK POSITIONS are unchanged so PK still finds the right (dy,dx).',
           _fn_cc, manual='CC'))
