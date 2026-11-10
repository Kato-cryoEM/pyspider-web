"""Built-in node — data.mo3d (MO 3D · Model volume) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_mo3d)

register(N('data.mo3d', 'Model volume', 'Data', '', 'volume:volume', [('nz', 'int', 128, None, 4, 512), ('ny', 'int', 128, None, 4, 512), ('nx', 'int', 128, None, 4, 512), ('kind', 'select', 'sphere', ['blank', 'sphere', 'gaussian', 'cube']), ('density', 'float', 1.0), ('radius', 'float', 8.0, None, 0.0, 512.0), ('center_z', 'float', -1.0), ('center_y', 'float', -1.0), ('center_x', 'float', -1.0)], '3D counterpart of MO — synthetic volume for reconstruction / projection demos. sphere / gaussian centre a shape of `radius` on (center_z, y, x); cube uses `radius` as half-side; blank fills with `density`. Center coordinates of -1 auto-use the volume centre.', _fn_mo3d))
