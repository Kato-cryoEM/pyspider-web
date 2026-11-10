"""Built-in node — ctf.model (TF C · CTF model) — category: CTF.

One plugin file per node so individual nodes can be enabled/disabled by
moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_CTF_PARAMS, _fn_ctf_model)

register(N(
    'ctf.model', 'CTF model', 'CTF', '', 'ctf:image',
    [('nx', 'int', 256, None, 32, 2048),
     ('ny', 'int', 256, None, 32, 2048)] + _CTF_PARAMS,
    'Build a 2D contrast transfer function on the **full (ny, nx) plane with '
    'DC pre-shifted to (ny // 2, nx // 2)** — same shape and centring as FT '
    'output, so the CTF can be wired directly into a MU alongside an FT\'d '
    'image (product → IFT for CTF-weighted filtering). '
    'voltage_kV / cs_mm / pixel_A describe the microscope; defocus_A is the '
    'underfocus (positive = under-focus); amp_contrast is the amplitude-to-'
    'phase contrast ratio; astigmatism_A / azimuth_deg add elliptic '
    'astigmatism; b_factor applies a Gaussian envelope exp(-B k² / 4) if > 0.',
    _fn_ctf_model, manual='TF C'))
