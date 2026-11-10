"""Built-in node — ft.amplitude (Amplitude spectrum) — category: Fourier.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_amplitude)

register(N('ft.amplitude', 'Amplitude spectrum', 'Fourier', 'fourier:fourier', 'image:image', [], 'Magnitude spectrum |F(k)|, with an fftshift on the ky axis so that the DC component sits at the vertical centre.', _fn_amplitude))
