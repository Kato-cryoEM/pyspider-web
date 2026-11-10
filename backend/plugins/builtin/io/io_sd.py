"""Built-in node — io.sd (SD · Write doc file) — category: I/O.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_sd)

register(N('io.sd', 'Write doc file', 'I/O', '', '', [('path', 'string', 'doc.txt'), ('key', 'int', 1), ('n_cols', 'int', 3, None, 1, 8), ('v1', 'float', 0.0), ('v2', 'float', 0.0), ('v3', 'float', 0.0), ('v4', 'float', 0.0), ('v5', 'float', 0.0), ('v6', 'float', 0.0), ('v7', 'float', 0.0), ('v8', 'float', 0.0), ('append', 'select', 'yes', ['yes', 'no'])], "Append one row 'key ncol v1 v2 … v_{n_cols}' to a text doc file. `n_cols` chooses how many columns are written (1–8); wire scalar outputs (from FS, PK, CC C, OR SH, AP SH…) into v1..vN. **Multi-scalar nodes**: e.g. OR SH emits angle_deg, dy, dx, peak, mirror — wire them into v1..v5 and set n_cols=5. Put the SD node inside a DO loop and wire `i` into `key` to save one row per pass. For per-file separation, use `io.save`'s path template `{i}` instead.", _fn_sd, manual='SD'))
