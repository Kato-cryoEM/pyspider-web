"""Built-in node — ft.power (Power spectrum (image→image)) — category: Fourier.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_power_spectrum)

register(N('ft.power', 'Power spectrum (image→image)', 'Fourier', 'image:image', 'image:image', [], 'Shortcut node: FFT the image, take |F|², apply log compression and fftshift. Equivalent to FT → log-amplitude of |F|².', _fn_power_spectrum))
