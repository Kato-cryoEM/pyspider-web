"""Built-in node — cls.kmeans (CL KM · K-means) — category: Classification.

Auto-migrated from build_catalog(). One plugin file per node so individual
nodes can be enabled/disabled by moving or deleting this single file.
"""

from __future__ import annotations

from backend.plugin_api import register, N, Param, NodeSpec, Port
from backend.nodes import (_fn_kmeans)

register(N('cls.kmeans', 'K-means', 'Classification', 'scores:image', 'labels:image', [('n_clusters', 'int', 4, None, 2, 64), ('n_init', 'int', 10, None, 1, 100), ('max_iter', 'int', 200, None, 10, 5000), ('seed', 'int', 0, None, 0, 100000)], "Lloyd's k-means on the PCA score vectors. n_init random restarts (keeps the best inertia); max_iter is the per-restart iteration cap; seed fixes the RNG for reproducible cluster assignments.", _fn_kmeans, manual='CL KM'))
