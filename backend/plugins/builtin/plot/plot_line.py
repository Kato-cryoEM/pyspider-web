"""Built-in node — plot.line (Plot · Line / Scatter / Bar / Stem).

Turns numeric text (CSV, whitespace-separated, one-value-per-line) into a
2D grayscale plot image that can be previewed in the Inspector, wired into
Save Image (PNG), or Save MRC (float32).
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_plot)

register(N(
    'plot.line', 'Plot · Line / Scatter / Bar', 'Plot',
    'data:string', 'image:image',
    [
        Param('plot_type', 'select', 'line',
              ['line', 'scatter', 'bar', 'stem']),
        Param('title', 'string', ''),
        Param('xlabel', 'string', ''),
        Param('ylabel', 'string', ''),
        Param('x_column', 'int', -1, None, -1, 64),
        Param('y_column', 'int', 0, None, 0, 64),
        Param('width', 'int', 480, None, 64, 4096),
        Param('height', 'int', 360, None, 64, 4096),
        Param('grid', 'bool', True),
        Param('logscale_x', 'bool', False),
        Param('logscale_y', 'bool', False),
        Param('marker', 'select', 'auto',
              ['auto', 'none', 'o', 's', 'x', '+', 'D', '^', 'v']),
    ],
    'Render numeric text as a 2D plot image via matplotlib. The `data` '
    'string is parsed one row per line, splitting on comma / whitespace / '
    'semicolon / "=" (so CSV, WSV, and even `k = v` logs all work); '
    'blank lines and lines starting with "#" are skipped. '
    'x_column = -1 uses the row index as X (handy for time-series text like '
    'SGD `loss_history`); otherwise pick that column of the parsed table. '
    'y_column selects the value column. plot_type = line / scatter / bar / '
    'stem. The output is a grayscale image (H × W float32), so it plugs '
    'directly into Save Image (PNG / JPEG / TIFF) or Save MRC.',
    _fn_plot))
