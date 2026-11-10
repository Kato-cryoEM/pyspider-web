"""Built-in node — ctf.phase_flip (TF CT · Phase flip) — category: CTF.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_CTF_PARAMS, _fn_phase_flip)

register(N('ctf.phase_flip', 'Phase flip', 'CTF', 'image:image', 'image:image', _CTF_PARAMS, 'Multiplies each Fourier coefficient by sign(CTF(k)), fixing phase errors without changing the amplitude envelope. Fast and robust, but does not restore lost amplitude.', _fn_phase_flip, manual='TF CT'))
