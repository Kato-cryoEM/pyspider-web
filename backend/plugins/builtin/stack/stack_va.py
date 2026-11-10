"""Built-in node — stack.va (VA · Volume statistics) — category: Stack.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_va)

register(N('stack.va', 'Volume statistics', 'Stack', 'volume:volume', 'min:scalar,max:scalar,mean:scalar,sigma:scalar,sum:scalar,nz:scalar,ny:scalar,nx:scalar', [], '3D counterpart of FS — compute min / max / mean / sigma / sum of a volume plus its (nz, ny, nx) dimensions. Feed a stack to inspect the reconstructed volume from BP 3F / BP CG.', _fn_va))
