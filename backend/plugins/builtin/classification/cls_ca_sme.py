"""Built-in node — cls.ca_sme (CA SME · Scree list of eigenvalues)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ca_sme)

register(N(
    'cls.ca_sme', 'Scree table', 'Classification',
    'eigenvalues:stack', 'scree_csv:string,top_sum:scalar',
    [Param('top_n', 'int', 10, None, 1, 256)],
    'List the top-N eigenvalues from CA S / CA SI as a doc '
    'file. Emits `index, value, fraction` per row — wire into `Plot · Line` '
    '(y_column=1 or 2) to visualise the scree curve.',
    _fn_ca_sme, manual='CA SME'))
