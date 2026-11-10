"""Built-in node — data.doc_text (Doc · Text table) — category: Data.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_doc_text)

register(N('data.doc_text', 'Doc · Text table', 'Data', '', 'doc:doc,n_rows:scalar,last_row:scalar', [('text', 'text', '# angles: phi, theta, psi\n0, 0, 0\n45, 0, 0\n90, 0, 0')], "Multi-line text table. One row per line, columns separated by commas or whitespace. Blank lines and lines starting with `#` are ignored. Feed `doc` into a Doc Row inside a DO loop. For zero-indexed row iteration (start=0, step=1), wire `last_row` into DO's `end` — the loop then runs exactly `n_rows` passes with i = 0 .. n_rows-1.", _fn_doc_text))
