from __future__ import annotations

import numpy as np


def build_neighbor_index(geometry, *, max_neighbors: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous neighbors, with nearest-centroid fallback for isolates."""
    n = len(geometry)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    mask = np.zeros((n, max_neighbors), dtype=bool)
    distances = np.zeros((n, max_neighbors), dtype=np.float32)
    centroids = geometry.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    spatial_index = geometry.sindex
    for row, polygon in enumerate(geometry.geometry):
        candidates = list(spatial_index.query(polygon, predicate="touches"))
        candidates = [candidate for candidate in candidates if candidate != row]
        if not candidates and n > 1:
            d = np.linalg.norm(coords - coords[row], axis=1)
            d[row] = np.inf
            candidates = [int(np.argmin(d))]
        candidates = sorted(candidates, key=lambda j: float(np.linalg.norm(coords[j] - coords[row])))[:max_neighbors]
        count = len(candidates)
        if count:
            indices[row, :count] = candidates
            mask[row, :count] = True
            distances[row, :count] = np.linalg.norm(coords[candidates] - coords[row], axis=1)
    return indices, mask, distances
