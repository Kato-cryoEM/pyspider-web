"""Built-in node — ctf.wiener (Wiener CTF correct) — category: CTF.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_CTF_PARAMS, _fn_wiener)

register(N('ctf.wiener', 'Wiener CTF correct', 'CTF', 'image:image', 'image:image', _CTF_PARAMS + [('snr', 'float', 1.0, None, 0.01, 1000.0)], 'Wiener filter inverse CTF: H / (|H|² + 1/SNR). Restores both phase and amplitude, damped where the signal-to-noise ratio is low. Larger snr = closer to a naive division (noisier).', _fn_wiener))
