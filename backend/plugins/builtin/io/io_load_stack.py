"""Built-in node — io.load_stack (Load Stack) — category: I/O.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_load_stack)

register(N(
    'io.load_stack', 'Load Stack', 'I/O', '', 'stack:stack',
    [Param('path', 'string', '',
           placeholder='/absolute/path/to/stack.mrc'),
     ('n_max', 'int', 32, None, 1, 10000)],
    'Read up to n_max frames from a 3D MRC file. The result is a 3D array '
    '(N, ny, nx) usable by Stack / Classification / BP 3F. A 2D input is '
    'wrapped as a length-1 stack.',
    _fn_load_stack))
