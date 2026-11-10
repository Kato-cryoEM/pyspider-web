"""Built-in node — ft.phase (Phase spectrum) — category: Fourier.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_phase_spec)

register(N('ft.phase', 'Phase spectrum', 'Fourier',
           'fourier:fourier', 'image:image', [],
           'Arg(F(k)) in radians. Wrapped to (-π, π]. Preview is grayscale '
           '(dark = -π, light = +π). Useful for CTF phase analysis and shift '
           'diagnostics. For a combined phase+amplitude colour preview, look '
           'at the raw output of `ft.forward`.',
           _fn_phase_spec))
