"""Built-in node — cls.pca (CA S · Factor analysis) — category: Classification.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_pca)

register(N('cls.pca', 'Factor analysis', 'Classification', 'stack:stack', 'scores:image,eigenimage:image', [('n_components', 'int', 8, None, 2, 128), ('method', 'select', 'pca', ['pca', 'coran', 'iterative', 'skip']), ('additive_constant', 'float', 0.0)], 'SPIDER CA S methods: pca (P) plain PCA via SVD; coran (C) correspondence analysis using the χ² metric (needs non-negative data — add additive_constant if the stack has negatives); iterative (I) and skip (S) currently alias to plain PCA. Returns an (N, n_components) score matrix and the first eigen-image reshaped back to (ny, nx) for preview.', _fn_pca, manual='CA S'))
