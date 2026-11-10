"""Built-in node — data.doc_row (Doc · Row) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_doc_row)

register(N('data.doc_row', 'Doc · Row', 'Data', 'doc:doc', 'v1:scalar,v2:scalar,v3:scalar,v4:scalar', [('row', 'int', 0, None, 0, 100000)], "Return columns 1-4 of the `row`-th line of a doc as scalars. Row is zero-indexed. Wire a DO container's `i` into `row` to pull one line per iteration. Missing cells default to 0.0.", _fn_doc_row))
