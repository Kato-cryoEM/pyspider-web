"""Built-in node — ctf.apply (Apply CTF) — category: CTF.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_CTF_PARAMS, _fn_apply_ctf)

register(N('ctf.apply', 'Apply CTF', 'CTF', 'image:image', 'image:image', _CTF_PARAMS, 'Multiply the input by the CTF built from these parameters — the forward image-formation model. Use it to add CTF effects to a clean simulated image.', _fn_apply_ctf))
