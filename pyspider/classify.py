"""2D classification primitives in pure NumPy.

This module mirrors the classification pipeline used by SPIDER's
``CL KM`` / ``CL HC`` operations.  In SPIDER, clustering operates on
*factor coordinates* produced by Correspondence Analysis (``CA S``);
here we expose each step explicitly so the underlying algorithm is
visible:

  ``flatten_stack``   reshape an image stack ``(N, ny, nx)`` to a feature
                      matrix ``(N, ny * nx)`` (the simplest "no PCA"
                      feature).
  ``pca``             singular-value-decomposition based principal
                      component analysis -- the linear-algebra core of
                      SPIDER's ``CA S``.
  ``kmeans``          Lloyd's algorithm with random restarts.  The same
                      algorithm as ``CL KM`` (SPIDER calls scikit-style
                      iterative refinement of centroids on factor
                      coordinates).
  ``class_averages``  compute mean images per class -- the natural
                      output of any 2D classifier in cryo-EM.

The functions are pure NumPy.  Tests under ``tests/test_classify.py``
verify them against the canonical SciPy / scikit-learn implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers


def flatten_stack(stack: np.ndarray) -> np.ndarray:
    """Reshape ``(N, ny, nx)`` -> ``(N, ny*nx)`` in row-major order."""
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3D (N, ny, nx), got shape {stack.shape}")
    n, ny, nx = stack.shape
    return stack.reshape(n, ny * nx).astype(np.float32, copy=False)


def unflatten_stack(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape ``(N, ny*nx)`` -> ``(N, ny, nx)``."""
    ny, nx = shape
    if flat.ndim != 2 or flat.shape[1] != ny * nx:
        raise ValueError(
            f"flat shape {flat.shape} inconsistent with image shape {shape}"
        )
    return flat.reshape(flat.shape[0], ny, nx).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# PCA


@dataclass
class PCAResult:
    mean: np.ndarray                # (D,)
    components: np.ndarray          # (k, D), each row a principal vector
    singular_values: np.ndarray     # (k,)
    scores: np.ndarray              # (N, k), aka factor coordinates

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        s2 = self.singular_values ** 2
        return s2 / s2.sum() if s2.sum() > 0 else np.zeros_like(s2)


def pca(features: np.ndarray, n_components: int) -> PCAResult:
    """PCA via centered SVD.

    Returns the top ``n_components`` principal directions, their singular
    values, and the per-sample scores.  This is the linear algebra core
    of SPIDER's ``CA S`` (Correspondence Analysis is PCA with a different
    metric; for image stacks the metric is Euclidean and CORAN reduces
    to PCA).

    Notes
    -----
    For ``N < D`` (more dimensions than samples, the cryo-EM case for
    typical ``ny * nx >> N``), we compute the SVD of the smaller
    ``(N, N)`` Gram matrix and reconstruct components from it -- vastly
    faster than SVD-ing ``(N, D)`` directly.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    n, d = features.shape
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    n_components = min(n_components, n, d)

    mean = features.mean(axis=0)
    centered = features - mean

    if n <= d:
        # Gram-matrix trick: compute (N x N) eigendecomposition.
        gram = centered @ centered.T            # (N, N)
        # Symmetric eigendecomposition (ascending eigenvalues).
        eigvals, eigvecs = np.linalg.eigh(gram)
        idx = np.argsort(eigvals)[::-1][:n_components]
        eigvals = np.clip(eigvals[idx], 0.0, None)
        eigvecs = eigvecs[:, idx]
        singular_values = np.sqrt(eigvals).astype(np.float32)
        # Project gram eigenvectors back to feature space.  Each
        # component = (1 / sigma) * X.T @ u_i.
        sigma_inv = np.where(singular_values > 1e-12,
                             1.0 / singular_values, 0.0)
        components = (eigvecs.T @ centered) * sigma_inv[:, None]
        scores = eigvecs * singular_values[None, :]
    else:
        # Standard SVD on (N, D).
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_components]
        singular_values = S[:n_components].astype(np.float32)
        scores = U[:, :n_components] * singular_values[None, :]

    return PCAResult(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        singular_values=singular_values,
        scores=scores.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# K-means (Lloyd's algorithm)


@dataclass
class KMeansResult:
    centroids: np.ndarray           # (k, D)
    labels: np.ndarray              # (N,) int
    inertia: float                  # sum of squared distances to centroid


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialisation (smarter than uniform random seeds)."""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    # First centroid: uniform random.
    centroids[0] = X[rng.integers(0, n)]
    # Subsequent centroids: proportional to squared distance from nearest.
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else None
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
        new_sq = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centroids


def kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 200,
    n_init: int = 10,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> KMeansResult:
    """Standard Lloyd's algorithm with k-means++ initialisation.

    Returns the best of ``n_init`` random restarts (lowest inertia).
    """
    X = np.ascontiguousarray(features, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"features must be 2D, got {X.shape}")
    n, d = X.shape
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")
    rng = np.random.default_rng(seed)

    best: KMeansResult | None = None
    for _ in range(n_init):
        centroids = _kmeans_pp_init(X, n_clusters, rng)
        prev_inertia = np.inf
        for _it in range(max_iter):
            # Assignment step.
            # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; the ||x||^2 term
            # is constant across centroids so we drop it.
            dots = X @ centroids.T                    # (N, k)
            cnorm = np.sum(centroids ** 2, axis=1)    # (k,)
            d2 = cnorm[None, :] - 2.0 * dots
            labels = np.argmin(d2, axis=1)

            # Update step.
            new_centroids = np.empty_like(centroids)
            for j in range(n_clusters):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    # Empty cluster -- re-seed from the farthest point.
                    far = int(np.argmax(d2[np.arange(n), labels]))
                    new_centroids[j] = X[far]

            inertia = float(((X - new_centroids[labels]) ** 2).sum())
            if abs(prev_inertia - inertia) < tol * max(1.0, prev_inertia):
                centroids = new_centroids
                break
            prev_inertia = inertia
            centroids = new_centroids

        run = KMeansResult(centroids=centroids,
                           labels=labels.astype(np.int32),
                           inertia=inertia)
        if best is None or run.inertia < best.inertia:
            best = run
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Class averages -- the canonical output of any 2D classifier in cryo-EM.


def class_averages(
    stack: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class mean images.

    Returns ``(averages, counts)`` where ``averages`` has shape
    ``(n_classes, ny, nx)`` and ``counts`` shape ``(n_classes,)``.
    Empty classes get a zero average and ``count = 0``.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3D, got {stack.shape}")
    if labels.ndim != 1 or labels.shape[0] != stack.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} doesn't match stack {stack.shape}"
        )
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    ny, nx = stack.shape[1:]
    avgs = np.zeros((n_classes, ny, nx), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)
    for j in range(n_classes):
        mask = labels == j
        c = int(mask.sum())
        counts[j] = c
        if c > 0:
            avgs[j] = stack[mask].mean(axis=0)
    return avgs, counts
