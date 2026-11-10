"""Built-in node — io.load (Load MRC) — category: I/O.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_load)

register(N(
    'io.load', 'Load MRC', 'I/O', '', 'image:image',
    [Param('path', 'string', '',
           placeholder='/absolute/path/to/image.mrc')],
    'Read a 2D image from disk (MRC format). The extension picks the reader '
    'via the format registry, so any other extension you have a registered '
    'reader for (.map / .ccp4 / plugins) also works — but MRC is the default.',
    _fn_load))
