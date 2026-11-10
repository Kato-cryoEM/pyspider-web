"""Built-in node — io.save (Save MRC) — category: I/O.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_save)

register(N(
    'io.save', 'Save MRC', 'I/O', 'image:image', '',
    [Param('path', 'string', '',
           placeholder='out.mrc (use {i} for DO iterations)'),
     ('title', 'string', ''),
     ('index', 'int', -1)],
    'Write the incoming image to disk in MRC format. The extension picks '
    'the writer via the format registry. **DO-loop tip**: put `{i}` in the '
    'path (e.g. `proj_{i}.mrc`) and wire the DO container\'s `i` into the '
    '`index` param — every iteration will save to a distinct file instead '
    'of overwriting the same one. `title` is a short header label that '
    'writers may store if they support it.',
    _fn_save))
