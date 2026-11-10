"""Built-in node — ft.log_amplitude (Log-amplitude spectrum) — category: Fourier.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_log_amplitude)

register(N('ft.log_amplitude', 'Log-amplitude spectrum', 'Fourier', 'fourier:fourier', 'image:image', [], 'Log(1 + |F(k)|) — compresses the dynamic range so structure at high frequencies becomes visible even when the DC term dominates.', _fn_log_amplitude))
