"""Built-in node — cls.ca_si (CA SI · Full factor analysis with scores)."""

from __future__ import annotations
from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_ca_s)

_OUT = ('eigenimages:stack,eigenvalues:stack,mean_image:image,'
        'scores_csv:string,explained_ratio:scalar')

register(N(
    'cls.ca_si', 'Factor analysis (full)', 'Classification',
    'stack:stack', _OUT,
    [Param('method', 'select', 'pca', ['pca', 'coran']),
     Param('n_factors', 'int', 10, None, 1, 256),
     Param('subtract_mean', 'bool', True,
           depends_on=('method', 'pca'))],
    'Full factor / PCA analysis on an image stack. '
    'method: **pca** (mean-centred SVD) or **coran** (Correspondence '
    'Analysis with Chi-square metric — canonical for non-negative image '
    'data). Returns the top `n_factors` eigenimages (stack), their '
    'eigenvalues, the stack mean image, per-frame scores as a CSV doc-file '
    '(wire into CA SR to reconstitute individual frames), and the '
    'cumulative variance explained. See also `cls.pca` for the shorter '
    'CA S variant.',
    _fn_ca_s, manual='CA SI'))
