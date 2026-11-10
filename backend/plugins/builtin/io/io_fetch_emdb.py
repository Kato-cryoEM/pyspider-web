"""Built-in node — io.fetch_emdb (Fetch · EMDB map) — category: I/O.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_fetch_emdb)

register(N(
    'io.fetch_emdb', 'Fetch · EMDB map', 'I/O', '', 'volume:volume',
    [
        Param('emdb_id', 'string', '6840',
              placeholder='6840  or  EMD-6840'),
        Param('box_size', 'int', 128, None, 0, 4096),
        Param('cache_dir', 'string', '',
              placeholder='~/.pyspider_emdb_cache (default)'),
    ],
    'Download an EMDB density map (MRC-format .map.gz) from the wwPDB / EBI / '
    'PDBj mirrors and return the volume. Accepts either a bare number '
    '(`1010`, `1884`) or the full accession (`EMD-1010`). The map is cached '
    'under `~/.pyspider_emdb_cache/` (or `cache_dir` if set), so re-runs are '
    'instant. '
    '**`box_size`**: target cubic dimension in pixels (0 = keep original). '
    'When the requested size is **smaller** than the source, the resize is '
    'done by **Fourier crop** — ideal low-pass + subsample, no aliasing. '
    'When it is **larger**, cubic B-spline (order = 3, the 3D analog of '
    'bicubic) interpolation is used. '
    'Requires internet on first fetch. Examples: EMD-1010 (GroEL, ~800 KB), '
    'EMD-1884 (ribosome), EMD-5502 (proteasome).',
    _fn_fetch_emdb))
